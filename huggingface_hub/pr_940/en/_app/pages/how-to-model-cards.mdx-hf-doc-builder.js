import{S as kr,i as Pr,s as Tr,e as r,k as c,w as u,t as l,M as Dr,c as o,d as e,m as h,a as i,x as m,h as n,b as d,G as s,g as p,y as f,L as Ar,q as g,o as _,B as v,v as qr}from"../chunks/vendor-hf-doc-builder.js";import{I as na}from"../chunks/IconCopyLink-hf-doc-builder.js";import{C as j}from"../chunks/CodeBlock-hf-doc-builder.js";function Rr(un){let P,We,T,N,ne,ra,pt,re,ct,ze,w,ht,oe,dt,ut,oa,mt,ft,Ve,D,L,ie,ia,gt,pe,_t,Ge,x,vt,Na,jt,yt,pa,ce,bt,$t,Ke,ca,Qe,La,wt,Xe,E,C,he,xt,Et,Oa,Ct,Mt,de,kt,Pt,Tt,O,ue,Dt,At,me,qt,Rt,St,F,fe,Ht,It,ge,Nt,Lt,Ze,A,B,_e,ha,Ot,ve,Ft,as,q,U,je,da,Bt,ye,Ut,es,J,Jt,be,Yt,Wt,ss,ua,ts,Fa,zt,ls,Y,ma,Vt,Ba,Gt,Kt,Qt,$e,Xt,ns,fa,rs,Ua,Zt,os,ga,is,R,W,we,_a,al,xe,el,ps,z,sl,Ee,tl,ll,cs,va,hs,Ja,nl,ds,ja,us,Ya,rl,ms,ya,fs,Wa,ol,gs,ba,_s,V,il,za,pl,cl,vs,S,G,Ce,$a,hl,Me,dl,js,K,ul,wa,ml,fl,ys,xa,bs,H,Q,ke,Ea,gl,Pe,_l,$s,b,vl,Te,jl,yl,De,bl,$l,Va,wl,xl,ws,Ga,El,xs,Ca,Es,Ka,Cl,Cs,Ma,Ms,Qa,Ml,ks,ka,Ps,X,kl,Pa,Pl,Tl,Ts,M,Dl,Ae,Al,ql,qe,Rl,Sl,Ds,Ta,As,Z,Hl,Da,Il,Nl,qs,I,aa,Re,Aa,Ll,Se,Ol,Rs,y,Fl,He,Bl,Ul,Xa,Jl,Yl,Ie,Wl,zl,Ne,Vl,Gl,Le,Kl,Ql,Ss,k,Xl,Oe,Zl,an,Za,en,sn,Hs,qa,Is,ea,tn,Fe,ln,nn,Ns,Ra,Ls,sa,rn,Be,on,pn,Os,Sa,Fs,ta,cn,Ue,hn,dn,Bs,Ha,Us;return ra=new na({}),ia=new na({}),ca=new j({props:{code:`from huggingface_hub import ModelCard

card = ModelCard.load('nateraw/vit-base-beans')`,highlighted:`<span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> ModelCard

card = ModelCard.load(<span class="hljs-string">&#x27;nateraw/vit-base-beans&#x27;</span>)`}}),ha=new na({}),da=new na({}),ua=new j({props:{code:`text = """
---
language: en
license: mit
---

# My Model Card
"""

card = ModelCard(text)
card.data.to_dict() == {'language': 'en', 'license': 'mit'}  # True`,highlighted:`text = <span class="hljs-string">&quot;&quot;&quot;
---
language: en
license: mit
---

# My Model Card
&quot;&quot;&quot;</span>

card = ModelCard(text)
card.data.to_dict() == {<span class="hljs-string">&#x27;language&#x27;</span>: <span class="hljs-string">&#x27;en&#x27;</span>, <span class="hljs-string">&#x27;license&#x27;</span>: <span class="hljs-string">&#x27;mit&#x27;</span>}  <span class="hljs-comment"># True</span>`}}),fa=new j({props:{code:`card_data = ModelCardData(language='en', license='mit', library='timm')

example_template_var = 'nateraw'
text = f"""
---
{ card_data.to_yaml() }
---

# My Model Card

This model created by [@{example_template_var}](https://github.com/{example_template_var})
"""

card = ModelCard(text)
print(card)`,highlighted:`card_data = ModelCardData(language=<span class="hljs-string">&#x27;en&#x27;</span>, license=<span class="hljs-string">&#x27;mit&#x27;</span>, library=<span class="hljs-string">&#x27;timm&#x27;</span>)

example_template_var = <span class="hljs-string">&#x27;nateraw&#x27;</span>
text = <span class="hljs-string">f&quot;&quot;&quot;
---
<span class="hljs-subst">{ card_data.to_yaml() }</span>
---

# My Model Card

This model created by [@<span class="hljs-subst">{example_template_var}</span>](https://github.com/<span class="hljs-subst">{example_template_var}</span>)
&quot;&quot;&quot;</span>

card = ModelCard(text)
<span class="hljs-built_in">print</span>(card)`}}),ga=new j({props:{code:`---
language: en
license: mit
library: timm
---

# My Model Card

This model created by [@nateraw](https://github.com/nateraw)`,highlighted:`<span class="hljs-meta">---</span>
<span class="hljs-attr">language:</span> <span class="hljs-string">en</span>
<span class="hljs-attr">license:</span> <span class="hljs-string">mit</span>
<span class="hljs-attr">library:</span> <span class="hljs-string">timm</span>
<span class="hljs-meta">---
</span>
<span class="hljs-comment"># My Model Card</span>

<span class="hljs-string">This</span> <span class="hljs-string">model</span> <span class="hljs-string">created</span> <span class="hljs-string">by</span> [<span class="hljs-string">@nateraw</span>]<span class="hljs-string">(https://github.com/nateraw)</span>`}}),_a=new na({}),va=new j({props:{code:`from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData

# Define your jinja template
template_text = """
---
{{ card_data }}
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@{{ author }}](https://hf.co/{{author}}).
""".strip()

# Write the template to a file
Path('custom_template.md').write_text(template_text)

# Define card metadata
card_data = ModelCardData(language='en', license='mit', library_name='keras')

# Create card from template, passing it any jinja template variables you want.
# In our case, we'll pass author
card = ModelCard.from_template(card_data, template_path='custom_template.md', author='nateraw')
card.save('my_model_card_1.md')
print(card)`,highlighted:`<span class="hljs-keyword">from</span> pathlib <span class="hljs-keyword">import</span> Path

<span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> ModelCard, ModelCardData

<span class="hljs-comment"># Define your jinja template</span>
template_text = <span class="hljs-string">&quot;&quot;&quot;
---
{{ card_data }}
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@{{ author }}](https://hf.co/{{author}}).
&quot;&quot;&quot;</span>.strip()

<span class="hljs-comment"># Write the template to a file</span>
Path(<span class="hljs-string">&#x27;custom_template.md&#x27;</span>).write_text(template_text)

<span class="hljs-comment"># Define card metadata</span>
card_data = ModelCardData(language=<span class="hljs-string">&#x27;en&#x27;</span>, license=<span class="hljs-string">&#x27;mit&#x27;</span>, library_name=<span class="hljs-string">&#x27;keras&#x27;</span>)

<span class="hljs-comment"># Create card from template, passing it any jinja template variables you want.</span>
<span class="hljs-comment"># In our case, we&#x27;ll pass author</span>
card = ModelCard.from_template(card_data, template_path=<span class="hljs-string">&#x27;custom_template.md&#x27;</span>, author=<span class="hljs-string">&#x27;nateraw&#x27;</span>)
card.save(<span class="hljs-string">&#x27;my_model_card_1.md&#x27;</span>)
<span class="hljs-built_in">print</span>(card)`}}),ja=new j({props:{code:`---
language: en
license: mit
library_name: keras
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@nateraw](https://hf.co/nateraw).`,highlighted:`<span class="hljs-meta">---</span>
<span class="hljs-attr">language:</span> <span class="hljs-string">en</span>
<span class="hljs-attr">license:</span> <span class="hljs-string">mit</span>
<span class="hljs-attr">library_name:</span> <span class="hljs-string">keras</span>
<span class="hljs-meta">---
</span>
<span class="hljs-comment"># Model Card for MyCoolModel</span>

<span class="hljs-string">This</span> <span class="hljs-string">model</span> <span class="hljs-string">does</span> <span class="hljs-string">this</span> <span class="hljs-string">and</span> <span class="hljs-string">that.</span>

<span class="hljs-string">This</span> <span class="hljs-string">model</span> <span class="hljs-string">was</span> <span class="hljs-string">created</span> <span class="hljs-string">by</span> [<span class="hljs-string">@nateraw</span>]<span class="hljs-string">(https://hf.co/nateraw).</span>`}}),ya=new j({props:{code:`card.data.library_name = 'timm'
card.data.language = 'fr'
card.data.license = 'apache-2.0'
print(card)`,highlighted:`card<span class="hljs-selector-class">.data</span><span class="hljs-selector-class">.library_name</span> = <span class="hljs-string">&#x27;timm&#x27;</span>
card<span class="hljs-selector-class">.data</span><span class="hljs-selector-class">.language</span> = <span class="hljs-string">&#x27;fr&#x27;</span>
card<span class="hljs-selector-class">.data</span><span class="hljs-selector-class">.license</span> = <span class="hljs-string">&#x27;apache-2.0&#x27;</span>
<span class="hljs-function"><span class="hljs-title">print</span><span class="hljs-params">(card)</span></span>`}}),ba=new j({props:{code:`---
language: fr
license: apache-2.0
library_name: timm
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@nateraw](https://hf.co/nateraw).`,highlighted:`<span class="hljs-meta">---</span>
<span class="hljs-attr">language:</span> <span class="hljs-string">fr</span>
<span class="hljs-attr">license:</span> <span class="hljs-string">apache-2.0</span>
<span class="hljs-attr">library_name:</span> <span class="hljs-string">timm</span>
<span class="hljs-meta">---
</span>
<span class="hljs-comment"># Model Card for MyCoolModel</span>

<span class="hljs-string">This</span> <span class="hljs-string">model</span> <span class="hljs-string">does</span> <span class="hljs-string">this</span> <span class="hljs-string">and</span> <span class="hljs-string">that.</span>

<span class="hljs-string">This</span> <span class="hljs-string">model</span> <span class="hljs-string">was</span> <span class="hljs-string">created</span> <span class="hljs-string">by</span> [<span class="hljs-string">@nateraw</span>]<span class="hljs-string">(https://hf.co/nateraw).</span>`}}),$a=new na({}),xa=new j({props:{code:`card_data = ModelCardData(language='en', license='mit', library_name='keras')
card = ModelCard.from_template(
    card_data,
    model_id='my-cool-model',
    model_description="this model does this and that",
    developers="Nate Raw",
    more_resources="https://github.com/huggingface/huggingface_hub",
    **card_data.to_dict()  # Pass along any card data vals that might fill out part of the template
)
card.save('my_model_card_2.md')
print(card)`,highlighted:`card_data = ModelCardData(language=<span class="hljs-string">&#x27;en&#x27;</span>, license=<span class="hljs-string">&#x27;mit&#x27;</span>, library_name=<span class="hljs-string">&#x27;keras&#x27;</span>)
card = ModelCard.from_template(
    card_data,
    model_id=<span class="hljs-string">&#x27;my-cool-model&#x27;</span>,
    model_description=<span class="hljs-string">&quot;this model does this and that&quot;</span>,
    developers=<span class="hljs-string">&quot;Nate Raw&quot;</span>,
    more_resources=<span class="hljs-string">&quot;https://github.com/huggingface/huggingface_hub&quot;</span>,
    **card_data.to_dict()  <span class="hljs-comment"># Pass along any card data vals that might fill out part of the template</span>
)
card.save(<span class="hljs-string">&#x27;my_model_card_2.md&#x27;</span>)
<span class="hljs-built_in">print</span>(card)`}}),Ea=new na({}),Ca=new j({props:{code:`from huggingface_hub import whoami, create_repo

user = whoami()['name']
repo_id = f'{user}/hf-hub-modelcards-pr-test'
url = create_repo(repo_id, exist_ok=True)`,highlighted:`<span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> whoami, create_repo

user = whoami()[<span class="hljs-string">&#x27;name&#x27;</span>]
repo_id = <span class="hljs-string">f&#x27;<span class="hljs-subst">{user}</span>/hf-hub-modelcards-pr-test&#x27;</span>
url = create_repo(repo_id, exist_ok=<span class="hljs-literal">True</span>)`}}),Ma=new j({props:{code:`card_data = ModelCardData(language='en', license='mit', library_name='keras')
card = ModelCard.from_template(
    card_data,
    model_id='my-cool-model',
    model_description="this model does this and that",
    developers="Nate Raw",
    more_resources="https://github.com/huggingface/huggingface_hub",
    **card_data.to_dict()  # Pass along any card data vals that might fill out part of the template
)`,highlighted:`card_data = ModelCardData(language=<span class="hljs-string">&#x27;en&#x27;</span>, license=<span class="hljs-string">&#x27;mit&#x27;</span>, library_name=<span class="hljs-string">&#x27;keras&#x27;</span>)
card = ModelCard.from_template(
    card_data,
    model_id=<span class="hljs-string">&#x27;my-cool-model&#x27;</span>,
    model_description=<span class="hljs-string">&quot;this model does this and that&quot;</span>,
    developers=<span class="hljs-string">&quot;Nate Raw&quot;</span>,
    more_resources=<span class="hljs-string">&quot;https://github.com/huggingface/huggingface_hub&quot;</span>,
    **card_data.to_dict()  <span class="hljs-comment"># Pass along any card data vals that might fill out part of the template</span>
)`}}),ka=new j({props:{code:"card.push_to_hub(repo_id)",highlighted:"card.push_to_hub(repo_id)"}}),Ta=new j({props:{code:"card.push_to_hub(repo_id, create_pr=True)",highlighted:'card.push_to_hub(repo_id, create_pr=<span class="hljs-literal">True</span>)'}}),Aa=new na({}),qa=new j({props:{code:`card_data = ModelCardData(
    language='en',
    license='mit',
    model_name='my-cool-model',
    eval_results = EvalResult(
        task_type='image-classification',
        dataset_type='beans',
        dataset_name='Beans',
        metric_type='accuracy',
        metric_value=0.7
    )
)

card = ModelCard.from_template(card_data)
print(card.data)`,highlighted:`card_data = ModelCardData(
    language=<span class="hljs-string">&#x27;en&#x27;</span>,
    license=<span class="hljs-string">&#x27;mit&#x27;</span>,
    model_name=<span class="hljs-string">&#x27;my-cool-model&#x27;</span>,
    eval_results = EvalResult(
        task_type=<span class="hljs-string">&#x27;image-classification&#x27;</span>,
        dataset_type=<span class="hljs-string">&#x27;beans&#x27;</span>,
        dataset_name=<span class="hljs-string">&#x27;Beans&#x27;</span>,
        metric_type=<span class="hljs-string">&#x27;accuracy&#x27;</span>,
        metric_value=<span class="hljs-number">0.7</span>
    )
)

card = ModelCard.from_template(card_data)
<span class="hljs-built_in">print</span>(card.data)`}}),Ra=new j({props:{code:`language: en
license: mit
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      name: Beans
      type: beans
    metrics:
    - type: accuracy
      value: 0.7`,highlighted:`<span class="hljs-attribute">language</span><span class="hljs-punctuation">:</span> <span class="hljs-string">en</span>
<span class="hljs-attribute">license</span><span class="hljs-punctuation">:</span> <span class="hljs-string">mit</span>
<span class="hljs-attribute">model-index</span><span class="hljs-punctuation">:</span>
<span class="hljs-bullet">-</span> <span class="hljs-string">name: my-cool-model</span>
  <span class="hljs-attribute">results</span><span class="hljs-punctuation">:</span>
  <span class="hljs-bullet">-</span> <span class="hljs-string">task:</span>
      <span class="hljs-attribute">type</span><span class="hljs-punctuation">:</span> <span class="hljs-string">image-classification</span>
    <span class="hljs-attribute">dataset</span><span class="hljs-punctuation">:</span>
      <span class="hljs-attribute">name</span><span class="hljs-punctuation">:</span> <span class="hljs-string">Beans</span>
      <span class="hljs-attribute">type</span><span class="hljs-punctuation">:</span> <span class="hljs-string">beans</span>
    <span class="hljs-attribute">metrics</span><span class="hljs-punctuation">:</span>
    <span class="hljs-bullet">-</span> <span class="hljs-string">type: accuracy</span>
      <span class="hljs-attribute">value</span><span class="hljs-punctuation">:</span> <span class="hljs-string">0.7</span>`}}),Sa=new j({props:{code:`card_data = ModelCardData(
    language='en',
    license='mit',
    model_name='my-cool-model',
    eval_results = [
        EvalResult(
            task_type='image-classification',
            dataset_type='beans',
            dataset_name='Beans',
            metric_type='accuracy',
            metric_value=0.7
        ),
        EvalResult(
            task_type='image-classification',
            dataset_type='beans',
            dataset_name='Beans',
            metric_type='f1',
            metric_value=0.65
        )
    ]
)
card = ModelCard.from_template(card_data)
card.data`,highlighted:`card_data = ModelCardData(
    language=<span class="hljs-string">&#x27;en&#x27;</span>,
    license=<span class="hljs-string">&#x27;mit&#x27;</span>,
    model_name=<span class="hljs-string">&#x27;my-cool-model&#x27;</span>,
    eval_results = [
        EvalResult(
            task_type=<span class="hljs-string">&#x27;image-classification&#x27;</span>,
            dataset_type=<span class="hljs-string">&#x27;beans&#x27;</span>,
            dataset_name=<span class="hljs-string">&#x27;Beans&#x27;</span>,
            metric_type=<span class="hljs-string">&#x27;accuracy&#x27;</span>,
            metric_value=<span class="hljs-number">0.7</span>
        ),
        EvalResult(
            task_type=<span class="hljs-string">&#x27;image-classification&#x27;</span>,
            dataset_type=<span class="hljs-string">&#x27;beans&#x27;</span>,
            dataset_name=<span class="hljs-string">&#x27;Beans&#x27;</span>,
            metric_type=<span class="hljs-string">&#x27;f1&#x27;</span>,
            metric_value=<span class="hljs-number">0.65</span>
        )
    ]
)
card = ModelCard.from_template(card_data)
card.data`}}),Ha=new j({props:{code:`language: en
license: mit
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      name: Beans
      type: beans
    metrics:
    - type: accuracy
      value: 0.7
    - type: f1
      value: 0.65`,highlighted:`<span class="hljs-attribute">language</span><span class="hljs-punctuation">:</span> <span class="hljs-string">en</span>
<span class="hljs-attribute">license</span><span class="hljs-punctuation">:</span> <span class="hljs-string">mit</span>
<span class="hljs-attribute">model-index</span><span class="hljs-punctuation">:</span>
<span class="hljs-bullet">-</span> <span class="hljs-string">name: my-cool-model</span>
  <span class="hljs-attribute">results</span><span class="hljs-punctuation">:</span>
  <span class="hljs-bullet">-</span> <span class="hljs-string">task:</span>
      <span class="hljs-attribute">type</span><span class="hljs-punctuation">:</span> <span class="hljs-string">image-classification</span>
    <span class="hljs-attribute">dataset</span><span class="hljs-punctuation">:</span>
      <span class="hljs-attribute">name</span><span class="hljs-punctuation">:</span> <span class="hljs-string">Beans</span>
      <span class="hljs-attribute">type</span><span class="hljs-punctuation">:</span> <span class="hljs-string">beans</span>
    <span class="hljs-attribute">metrics</span><span class="hljs-punctuation">:</span>
    <span class="hljs-bullet">-</span> <span class="hljs-string">type: accuracy</span>
      <span class="hljs-attribute">value</span><span class="hljs-punctuation">:</span> <span class="hljs-string">0.7</span>
    <span class="hljs-bullet">-</span> <span class="hljs-string">type: f1</span>
      <span class="hljs-attribute">value</span><span class="hljs-punctuation">:</span> <span class="hljs-string">0.65</span>`}}),{c(){P=r("meta"),We=c(),T=r("h1"),N=r("a"),ne=r("span"),u(ra.$$.fragment),pt=c(),re=r("span"),ct=l("Creating and Sharing Model Cards"),ze=c(),w=r("p"),ht=l("The "),oe=r("code"),dt=l("huggingface_hub"),ut=l(` library provides a Python interface to create, share, and update Model Cards.
Visit `),oa=r("a"),mt=l("the dedicated documentation page"),ft=l(`
for a deeper view of what Model Cards on the Hub are, and how they work under the hood.`),Ve=c(),D=r("h2"),L=r("a"),ie=r("span"),u(ia.$$.fragment),gt=c(),pe=r("span"),_t=l("Loading a Model Card from the Hub"),Ge=c(),x=r("p"),vt=l("To load an existing card from the hub, you can use the "),Na=r("a"),jt=l("ModelCard.load()"),yt=l(" function. Here, we\u2019ll load the card from "),pa=r("a"),ce=r("code"),bt=l("nateraw/vit-base-beans"),$t=l("."),Ke=c(),u(ca.$$.fragment),Qe=c(),La=r("p"),wt=l("This card has some helpful attributes that you may want to access/leverage:"),Xe=c(),E=r("ul"),C=r("li"),he=r("code"),xt=l("card.data"),Et=l(": Returns a "),Oa=r("a"),Ct=l("ModelCardData"),Mt=l(" instance with the model card\u2019s metadata. Call "),de=r("code"),kt=l(".to_dict()"),Pt=l(" on this instance to get the representation as a dictionary."),Tt=c(),O=r("li"),ue=r("code"),Dt=l("card.text"),At=l(": Returns the text of the card, "),me=r("em"),qt=l("excluding the metadata header"),Rt=l("."),St=c(),F=r("li"),fe=r("code"),Ht=l("card.content"),It=l(": Returns the text content of the card, "),ge=r("em"),Nt=l("including the metadata header"),Lt=l("."),Ze=c(),A=r("h2"),B=r("a"),_e=r("span"),u(ha.$$.fragment),Ot=c(),ve=r("span"),Ft=l("Creating Model Cards"),as=c(),q=r("h3"),U=r("a"),je=r("span"),u(da.$$.fragment),Bt=c(),ye=r("span"),Ut=l("From Text"),es=c(),J=r("p"),Jt=l("To initialize a Model Card from text, just pass the text content of the card to the "),be=r("code"),Yt=l("ModelCard"),Wt=l(" on init."),ss=c(),u(ua.$$.fragment),ts=c(),Fa=r("p"),zt=l("Another way you might want to do this is with f-strings. In the following example, we:"),ls=c(),Y=r("ul"),ma=r("li"),Vt=l("Use "),Ba=r("a"),Gt=l("ModelCardData.to_yaml()"),Kt=l(" to convert metadata we defined to YAML so we can use it to insert the YAML block in the model card"),Qt=c(),$e=r("li"),Xt=l("Show how you might use a template variable via Python f-strings."),ns=c(),u(fa.$$.fragment),rs=c(),Ua=r("p"),Zt=l("The above example would leave us with a card that looks like this:"),os=c(),u(ga.$$.fragment),is=c(),R=r("h3"),W=r("a"),we=r("span"),u(_a.$$.fragment),al=c(),xe=r("span"),el=l("From a Jinja Template"),ps=c(),z=r("p"),sl=l("If you have "),Ee=r("code"),tl=l("Jinja2"),ll=l(" installed, you can create Model Cards from a jinja template file. Let\u2019s see a basic example:"),cs=c(),u(va.$$.fragment),hs=c(),Ja=r("p"),nl=l("The resulting card\u2019s markdown looks like this:"),ds=c(),u(ja.$$.fragment),us=c(),Ya=r("p"),rl=l("If you update any card.data, it\u2019ll reflect in the card itself."),ms=c(),u(ya.$$.fragment),fs=c(),Wa=r("p"),ol=l("Now, as you can see, the metadata header has been updated:"),gs=c(),u(ba.$$.fragment),_s=c(),V=r("p"),il=l("As you update the card data, you can validate the card is still valid against the Hub by calling "),za=r("a"),pl=l("ModelCard.validate()"),cl=l(". This ensures that the card passes any validation rules set up on the Hugging Face Hub."),vs=c(),S=r("h3"),G=r("a"),Ce=r("span"),u($a.$$.fragment),hl=c(),Me=r("span"),dl=l("From the Default Template"),js=c(),K=r("p"),ul=l("Instead of using your own template, you can also use the "),wa=r("a"),ml=l("default template"),fl=l(", which is a fully featured model card with tons of sections you may want to fill out."),ys=c(),u(xa.$$.fragment),bs=c(),H=r("h2"),Q=r("a"),ke=r("span"),u(Ea.$$.fragment),gl=c(),Pe=r("span"),_l=l("Sharing Model Cards"),$s=c(),b=r("p"),vl=l("If you\u2019re authenticated with the Hugging Face Hub (either by using "),Te=r("code"),jl=l("huggingface-cli login"),yl=l(" or "),De=r("code"),bl=l("huggingface_hub.notebook_login()"),$l=l("), you can push cards to hub by simply calling "),Va=r("a"),wl=l("ModelCard.push_to_hub()"),xl=l(". Let\u2019s take a look at how to do that\u2026"),ws=c(),Ga=r("p"),El=l("First, we\u2019ll create a new repo called \u2018hf-hub-modelcards-pr-test\u2019 under the under the authenticated user\u2019s namespace:"),xs=c(),u(Ca.$$.fragment),Es=c(),Ka=r("p"),Cl=l("Then, we\u2019ll create a card from the default template (same as the one defined in the section above):"),Cs=c(),u(Ma.$$.fragment),Ms=c(),Qa=r("p"),Ml=l("Finally, we\u2019ll push that up to the hub"),ks=c(),u(ka.$$.fragment),Ps=c(),X=r("p"),kl=l("You can check out the resulting card "),Pa=r("a"),Pl=l("here"),Tl=l("."),Ts=c(),M=r("p"),Dl=l("If you instead wanted to push a card as a pull request, you can just say "),Ae=r("code"),Al=l("create_pr=True"),ql=l(" when calling "),qe=r("code"),Rl=l("push_to_hub"),Sl=l(":"),Ds=c(),u(Ta.$$.fragment),As=c(),Z=r("p"),Hl=l("A resulting PR created from this command can be seen "),Da=r("a"),Il=l("here"),Nl=l("."),qs=c(),I=r("h3"),aa=r("a"),Re=r("span"),u(Aa.$$.fragment),Ll=c(),Se=r("span"),Ol=l("Including Evaluation Results"),Rs=c(),y=r("p"),Fl=l("To include evaluation results in the metadata "),He=r("code"),Bl=l("model-index"),Ul=l(", you can pass an "),Xa=r("a"),Jl=l("EvalResult"),Yl=l(" or a list of "),Ie=r("code"),Wl=l("EvalResult"),zl=l(" with your associated evaluation results. Under the hood it\u2019ll create the "),Ne=r("code"),Vl=l("model-index"),Gl=l(" when you call "),Le=r("code"),Kl=l("card.data.to_dict()"),Ql=l("."),Ss=c(),k=r("p"),Xl=l("Note that it requires you to include the "),Oe=r("code"),Zl=l("model_name"),an=l(" attribute in "),Za=r("a"),en=l("ModelCardData"),sn=l("."),Hs=c(),u(qa.$$.fragment),Is=c(),ea=r("p"),tn=l("The resulting "),Fe=r("code"),ln=l("card.data"),nn=l(" should look like this:"),Ns=c(),u(Ra.$$.fragment),Ls=c(),sa=r("p"),rn=l("If you have more than one evaluation result you\u2019d like to share, just pass a list of "),Be=r("code"),on=l("EvalResult"),pn=l(":"),Os=c(),u(Sa.$$.fragment),Fs=c(),ta=r("p"),cn=l("Which should leave you with the following "),Ue=r("code"),hn=l("card.data"),dn=l(":"),Bs=c(),u(Ha.$$.fragment),this.h()},l(a){const t=Dr('[data-svelte="svelte-1phssyn"]',document.head);P=o(t,"META",{name:!0,content:!0}),t.forEach(e),We=h(a),T=o(a,"H1",{class:!0});var Js=i(T);N=o(Js,"A",{id:!0,class:!0,href:!0});var mn=i(N);ne=o(mn,"SPAN",{});var fn=i(ne);m(ra.$$.fragment,fn),fn.forEach(e),mn.forEach(e),pt=h(Js),re=o(Js,"SPAN",{});var gn=i(re);ct=n(gn,"Creating and Sharing Model Cards"),gn.forEach(e),Js.forEach(e),ze=h(a),w=o(a,"P",{});var ae=i(w);ht=n(ae,"The "),oe=o(ae,"CODE",{});var _n=i(oe);dt=n(_n,"huggingface_hub"),_n.forEach(e),ut=n(ae,` library provides a Python interface to create, share, and update Model Cards.
Visit `),oa=o(ae,"A",{href:!0,rel:!0});var vn=i(oa);mt=n(vn,"the dedicated documentation page"),vn.forEach(e),ft=n(ae,`
for a deeper view of what Model Cards on the Hub are, and how they work under the hood.`),ae.forEach(e),Ve=h(a),D=o(a,"H2",{class:!0});var Ys=i(D);L=o(Ys,"A",{id:!0,class:!0,href:!0});var jn=i(L);ie=o(jn,"SPAN",{});var yn=i(ie);m(ia.$$.fragment,yn),yn.forEach(e),jn.forEach(e),gt=h(Ys),pe=o(Ys,"SPAN",{});var bn=i(pe);_t=n(bn,"Loading a Model Card from the Hub"),bn.forEach(e),Ys.forEach(e),Ge=h(a),x=o(a,"P",{});var ee=i(x);vt=n(ee,"To load an existing card from the hub, you can use the "),Na=o(ee,"A",{href:!0});var $n=i(Na);jt=n($n,"ModelCard.load()"),$n.forEach(e),yt=n(ee," function. Here, we\u2019ll load the card from "),pa=o(ee,"A",{href:!0,rel:!0});var wn=i(pa);ce=o(wn,"CODE",{});var xn=i(ce);bt=n(xn,"nateraw/vit-base-beans"),xn.forEach(e),wn.forEach(e),$t=n(ee,"."),ee.forEach(e),Ke=h(a),m(ca.$$.fragment,a),Qe=h(a),La=o(a,"P",{});var En=i(La);wt=n(En,"This card has some helpful attributes that you may want to access/leverage:"),En.forEach(e),Xe=h(a),E=o(a,"UL",{});var se=i(E);C=o(se,"LI",{});var Ia=i(C);he=o(Ia,"CODE",{});var Cn=i(he);xt=n(Cn,"card.data"),Cn.forEach(e),Et=n(Ia,": Returns a "),Oa=o(Ia,"A",{href:!0});var Mn=i(Oa);Ct=n(Mn,"ModelCardData"),Mn.forEach(e),Mt=n(Ia," instance with the model card\u2019s metadata. Call "),de=o(Ia,"CODE",{});var kn=i(de);kt=n(kn,".to_dict()"),kn.forEach(e),Pt=n(Ia," on this instance to get the representation as a dictionary."),Ia.forEach(e),Tt=h(se),O=o(se,"LI",{});var Je=i(O);ue=o(Je,"CODE",{});var Pn=i(ue);Dt=n(Pn,"card.text"),Pn.forEach(e),At=n(Je,": Returns the text of the card, "),me=o(Je,"EM",{});var Tn=i(me);qt=n(Tn,"excluding the metadata header"),Tn.forEach(e),Rt=n(Je,"."),Je.forEach(e),St=h(se),F=o(se,"LI",{});var Ye=i(F);fe=o(Ye,"CODE",{});var Dn=i(fe);Ht=n(Dn,"card.content"),Dn.forEach(e),It=n(Ye,": Returns the text content of the card, "),ge=o(Ye,"EM",{});var An=i(ge);Nt=n(An,"including the metadata header"),An.forEach(e),Lt=n(Ye,"."),Ye.forEach(e),se.forEach(e),Ze=h(a),A=o(a,"H2",{class:!0});var Ws=i(A);B=o(Ws,"A",{id:!0,class:!0,href:!0});var qn=i(B);_e=o(qn,"SPAN",{});var Rn=i(_e);m(ha.$$.fragment,Rn),Rn.forEach(e),qn.forEach(e),Ot=h(Ws),ve=o(Ws,"SPAN",{});var Sn=i(ve);Ft=n(Sn,"Creating Model Cards"),Sn.forEach(e),Ws.forEach(e),as=h(a),q=o(a,"H3",{class:!0});var zs=i(q);U=o(zs,"A",{id:!0,class:!0,href:!0});var Hn=i(U);je=o(Hn,"SPAN",{});var In=i(je);m(da.$$.fragment,In),In.forEach(e),Hn.forEach(e),Bt=h(zs),ye=o(zs,"SPAN",{});var Nn=i(ye);Ut=n(Nn,"From Text"),Nn.forEach(e),zs.forEach(e),es=h(a),J=o(a,"P",{});var Vs=i(J);Jt=n(Vs,"To initialize a Model Card from text, just pass the text content of the card to the "),be=o(Vs,"CODE",{});var Ln=i(be);Yt=n(Ln,"ModelCard"),Ln.forEach(e),Wt=n(Vs," on init."),Vs.forEach(e),ss=h(a),m(ua.$$.fragment,a),ts=h(a),Fa=o(a,"P",{});var On=i(Fa);zt=n(On,"Another way you might want to do this is with f-strings. In the following example, we:"),On.forEach(e),ls=h(a),Y=o(a,"UL",{});var Gs=i(Y);ma=o(Gs,"LI",{});var Ks=i(ma);Vt=n(Ks,"Use "),Ba=o(Ks,"A",{href:!0});var Fn=i(Ba);Gt=n(Fn,"ModelCardData.to_yaml()"),Fn.forEach(e),Kt=n(Ks," to convert metadata we defined to YAML so we can use it to insert the YAML block in the model card"),Ks.forEach(e),Qt=h(Gs),$e=o(Gs,"LI",{});var Bn=i($e);Xt=n(Bn,"Show how you might use a template variable via Python f-strings."),Bn.forEach(e),Gs.forEach(e),ns=h(a),m(fa.$$.fragment,a),rs=h(a),Ua=o(a,"P",{});var Un=i(Ua);Zt=n(Un,"The above example would leave us with a card that looks like this:"),Un.forEach(e),os=h(a),m(ga.$$.fragment,a),is=h(a),R=o(a,"H3",{class:!0});var Qs=i(R);W=o(Qs,"A",{id:!0,class:!0,href:!0});var Jn=i(W);we=o(Jn,"SPAN",{});var Yn=i(we);m(_a.$$.fragment,Yn),Yn.forEach(e),Jn.forEach(e),al=h(Qs),xe=o(Qs,"SPAN",{});var Wn=i(xe);el=n(Wn,"From a Jinja Template"),Wn.forEach(e),Qs.forEach(e),ps=h(a),z=o(a,"P",{});var Xs=i(z);sl=n(Xs,"If you have "),Ee=o(Xs,"CODE",{});var zn=i(Ee);tl=n(zn,"Jinja2"),zn.forEach(e),ll=n(Xs," installed, you can create Model Cards from a jinja template file. Let\u2019s see a basic example:"),Xs.forEach(e),cs=h(a),m(va.$$.fragment,a),hs=h(a),Ja=o(a,"P",{});var Vn=i(Ja);nl=n(Vn,"The resulting card\u2019s markdown looks like this:"),Vn.forEach(e),ds=h(a),m(ja.$$.fragment,a),us=h(a),Ya=o(a,"P",{});var Gn=i(Ya);rl=n(Gn,"If you update any card.data, it\u2019ll reflect in the card itself."),Gn.forEach(e),ms=h(a),m(ya.$$.fragment,a),fs=h(a),Wa=o(a,"P",{});var Kn=i(Wa);ol=n(Kn,"Now, as you can see, the metadata header has been updated:"),Kn.forEach(e),gs=h(a),m(ba.$$.fragment,a),_s=h(a),V=o(a,"P",{});var Zs=i(V);il=n(Zs,"As you update the card data, you can validate the card is still valid against the Hub by calling "),za=o(Zs,"A",{href:!0});var Qn=i(za);pl=n(Qn,"ModelCard.validate()"),Qn.forEach(e),cl=n(Zs,". This ensures that the card passes any validation rules set up on the Hugging Face Hub."),Zs.forEach(e),vs=h(a),S=o(a,"H3",{class:!0});var at=i(S);G=o(at,"A",{id:!0,class:!0,href:!0});var Xn=i(G);Ce=o(Xn,"SPAN",{});var Zn=i(Ce);m($a.$$.fragment,Zn),Zn.forEach(e),Xn.forEach(e),hl=h(at),Me=o(at,"SPAN",{});var ar=i(Me);dl=n(ar,"From the Default Template"),ar.forEach(e),at.forEach(e),js=h(a),K=o(a,"P",{});var et=i(K);ul=n(et,"Instead of using your own template, you can also use the "),wa=o(et,"A",{href:!0,rel:!0});var er=i(wa);ml=n(er,"default template"),er.forEach(e),fl=n(et,", which is a fully featured model card with tons of sections you may want to fill out."),et.forEach(e),ys=h(a),m(xa.$$.fragment,a),bs=h(a),H=o(a,"H2",{class:!0});var st=i(H);Q=o(st,"A",{id:!0,class:!0,href:!0});var sr=i(Q);ke=o(sr,"SPAN",{});var tr=i(ke);m(Ea.$$.fragment,tr),tr.forEach(e),sr.forEach(e),gl=h(st),Pe=o(st,"SPAN",{});var lr=i(Pe);_l=n(lr,"Sharing Model Cards"),lr.forEach(e),st.forEach(e),$s=h(a),b=o(a,"P",{});var la=i(b);vl=n(la,"If you\u2019re authenticated with the Hugging Face Hub (either by using "),Te=o(la,"CODE",{});var nr=i(Te);jl=n(nr,"huggingface-cli login"),nr.forEach(e),yl=n(la," or "),De=o(la,"CODE",{});var rr=i(De);bl=n(rr,"huggingface_hub.notebook_login()"),rr.forEach(e),$l=n(la,"), you can push cards to hub by simply calling "),Va=o(la,"A",{href:!0});var or=i(Va);wl=n(or,"ModelCard.push_to_hub()"),or.forEach(e),xl=n(la,". Let\u2019s take a look at how to do that\u2026"),la.forEach(e),ws=h(a),Ga=o(a,"P",{});var ir=i(Ga);El=n(ir,"First, we\u2019ll create a new repo called \u2018hf-hub-modelcards-pr-test\u2019 under the under the authenticated user\u2019s namespace:"),ir.forEach(e),xs=h(a),m(Ca.$$.fragment,a),Es=h(a),Ka=o(a,"P",{});var pr=i(Ka);Cl=n(pr,"Then, we\u2019ll create a card from the default template (same as the one defined in the section above):"),pr.forEach(e),Cs=h(a),m(Ma.$$.fragment,a),Ms=h(a),Qa=o(a,"P",{});var cr=i(Qa);Ml=n(cr,"Finally, we\u2019ll push that up to the hub"),cr.forEach(e),ks=h(a),m(ka.$$.fragment,a),Ps=h(a),X=o(a,"P",{});var tt=i(X);kl=n(tt,"You can check out the resulting card "),Pa=o(tt,"A",{href:!0,rel:!0});var hr=i(Pa);Pl=n(hr,"here"),hr.forEach(e),Tl=n(tt,"."),tt.forEach(e),Ts=h(a),M=o(a,"P",{});var te=i(M);Dl=n(te,"If you instead wanted to push a card as a pull request, you can just say "),Ae=o(te,"CODE",{});var dr=i(Ae);Al=n(dr,"create_pr=True"),dr.forEach(e),ql=n(te," when calling "),qe=o(te,"CODE",{});var ur=i(qe);Rl=n(ur,"push_to_hub"),ur.forEach(e),Sl=n(te,":"),te.forEach(e),Ds=h(a),m(Ta.$$.fragment,a),As=h(a),Z=o(a,"P",{});var lt=i(Z);Hl=n(lt,"A resulting PR created from this command can be seen "),Da=o(lt,"A",{href:!0,rel:!0});var mr=i(Da);Il=n(mr,"here"),mr.forEach(e),Nl=n(lt,"."),lt.forEach(e),qs=h(a),I=o(a,"H3",{class:!0});var nt=i(I);aa=o(nt,"A",{id:!0,class:!0,href:!0});var fr=i(aa);Re=o(fr,"SPAN",{});var gr=i(Re);m(Aa.$$.fragment,gr),gr.forEach(e),fr.forEach(e),Ll=h(nt),Se=o(nt,"SPAN",{});var _r=i(Se);Ol=n(_r,"Including Evaluation Results"),_r.forEach(e),nt.forEach(e),Rs=h(a),y=o(a,"P",{});var $=i(y);Fl=n($,"To include evaluation results in the metadata "),He=o($,"CODE",{});var vr=i(He);Bl=n(vr,"model-index"),vr.forEach(e),Ul=n($,", you can pass an "),Xa=o($,"A",{href:!0});var jr=i(Xa);Jl=n(jr,"EvalResult"),jr.forEach(e),Yl=n($," or a list of "),Ie=o($,"CODE",{});var yr=i(Ie);Wl=n(yr,"EvalResult"),yr.forEach(e),zl=n($," with your associated evaluation results. Under the hood it\u2019ll create the "),Ne=o($,"CODE",{});var br=i(Ne);Vl=n(br,"model-index"),br.forEach(e),Gl=n($," when you call "),Le=o($,"CODE",{});var $r=i(Le);Kl=n($r,"card.data.to_dict()"),$r.forEach(e),Ql=n($,"."),$.forEach(e),Ss=h(a),k=o(a,"P",{});var le=i(k);Xl=n(le,"Note that it requires you to include the "),Oe=o(le,"CODE",{});var wr=i(Oe);Zl=n(wr,"model_name"),wr.forEach(e),an=n(le," attribute in "),Za=o(le,"A",{href:!0});var xr=i(Za);en=n(xr,"ModelCardData"),xr.forEach(e),sn=n(le,"."),le.forEach(e),Hs=h(a),m(qa.$$.fragment,a),Is=h(a),ea=o(a,"P",{});var rt=i(ea);tn=n(rt,"The resulting "),Fe=o(rt,"CODE",{});var Er=i(Fe);ln=n(Er,"card.data"),Er.forEach(e),nn=n(rt," should look like this:"),rt.forEach(e),Ns=h(a),m(Ra.$$.fragment,a),Ls=h(a),sa=o(a,"P",{});var ot=i(sa);rn=n(ot,"If you have more than one evaluation result you\u2019d like to share, just pass a list of "),Be=o(ot,"CODE",{});var Cr=i(Be);on=n(Cr,"EvalResult"),Cr.forEach(e),pn=n(ot,":"),ot.forEach(e),Os=h(a),m(Sa.$$.fragment,a),Fs=h(a),ta=o(a,"P",{});var it=i(ta);cn=n(it,"Which should leave you with the following "),Ue=o(it,"CODE",{});var Mr=i(Ue);hn=n(Mr,"card.data"),Mr.forEach(e),dn=n(it,":"),it.forEach(e),Bs=h(a),m(Ha.$$.fragment,a),this.h()},h(){d(P,"name","hf:doc:metadata"),d(P,"content",JSON.stringify(Sr)),d(N,"id","creating-and-sharing-model-cards"),d(N,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),d(N,"href","#creating-and-sharing-model-cards"),d(T,"class","relative group"),d(oa,"href","https://huggingface.co/docs/hub/models-cards"),d(oa,"rel","nofollow"),d(L,"id","loading-a-model-card-from-the-hub"),d(L,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),d(L,"href","#loading-a-model-card-from-the-hub"),d(D,"class","relative group"),d(Na,"href","/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.repocard.RepoCard.load"),d(pa,"href","https://huggingface.co/nateraw/vit-base-beans"),d(pa,"rel","nofollow"),d(Oa,"href","/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.ModelCardData"),d(B,"id","creating-model-cards"),d(B,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),d(B,"href","#creating-model-cards"),d(A,"class","relative group"),d(U,"id","from-text"),d(U,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),d(U,"href","#from-text"),d(q,"class","relative group"),d(Ba,"href","/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.CardData.to_yaml"),d(W,"id","from-a-jinja-template"),d(W,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),d(W,"href","#from-a-jinja-template"),d(R,"class","relative group"),d(za,"href","/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.repocard.RepoCard.validate"),d(G,"id","from-the-default-template"),d(G,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),d(G,"href","#from-the-default-template"),d(S,"class","relative group"),d(wa,"href","https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md"),d(wa,"rel","nofollow"),d(Q,"id","sharing-model-cards"),d(Q,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),d(Q,"href","#sharing-model-cards"),d(H,"class","relative group"),d(Va,"href","/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.repocard.RepoCard.push_to_hub"),d(Pa,"href","https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/blob/main/README.md"),d(Pa,"rel","nofollow"),d(Da,"href","https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/discussions/3"),d(Da,"rel","nofollow"),d(aa,"id","including-evaluation-results"),d(aa,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),d(aa,"href","#including-evaluation-results"),d(I,"class","relative group"),d(Xa,"href","/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.EvalResult"),d(Za,"href","/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.ModelCardData")},m(a,t){s(document.head,P),p(a,We,t),p(a,T,t),s(T,N),s(N,ne),f(ra,ne,null),s(T,pt),s(T,re),s(re,ct),p(a,ze,t),p(a,w,t),s(w,ht),s(w,oe),s(oe,dt),s(w,ut),s(w,oa),s(oa,mt),s(w,ft),p(a,Ve,t),p(a,D,t),s(D,L),s(L,ie),f(ia,ie,null),s(D,gt),s(D,pe),s(pe,_t),p(a,Ge,t),p(a,x,t),s(x,vt),s(x,Na),s(Na,jt),s(x,yt),s(x,pa),s(pa,ce),s(ce,bt),s(x,$t),p(a,Ke,t),f(ca,a,t),p(a,Qe,t),p(a,La,t),s(La,wt),p(a,Xe,t),p(a,E,t),s(E,C),s(C,he),s(he,xt),s(C,Et),s(C,Oa),s(Oa,Ct),s(C,Mt),s(C,de),s(de,kt),s(C,Pt),s(E,Tt),s(E,O),s(O,ue),s(ue,Dt),s(O,At),s(O,me),s(me,qt),s(O,Rt),s(E,St),s(E,F),s(F,fe),s(fe,Ht),s(F,It),s(F,ge),s(ge,Nt),s(F,Lt),p(a,Ze,t),p(a,A,t),s(A,B),s(B,_e),f(ha,_e,null),s(A,Ot),s(A,ve),s(ve,Ft),p(a,as,t),p(a,q,t),s(q,U),s(U,je),f(da,je,null),s(q,Bt),s(q,ye),s(ye,Ut),p(a,es,t),p(a,J,t),s(J,Jt),s(J,be),s(be,Yt),s(J,Wt),p(a,ss,t),f(ua,a,t),p(a,ts,t),p(a,Fa,t),s(Fa,zt),p(a,ls,t),p(a,Y,t),s(Y,ma),s(ma,Vt),s(ma,Ba),s(Ba,Gt),s(ma,Kt),s(Y,Qt),s(Y,$e),s($e,Xt),p(a,ns,t),f(fa,a,t),p(a,rs,t),p(a,Ua,t),s(Ua,Zt),p(a,os,t),f(ga,a,t),p(a,is,t),p(a,R,t),s(R,W),s(W,we),f(_a,we,null),s(R,al),s(R,xe),s(xe,el),p(a,ps,t),p(a,z,t),s(z,sl),s(z,Ee),s(Ee,tl),s(z,ll),p(a,cs,t),f(va,a,t),p(a,hs,t),p(a,Ja,t),s(Ja,nl),p(a,ds,t),f(ja,a,t),p(a,us,t),p(a,Ya,t),s(Ya,rl),p(a,ms,t),f(ya,a,t),p(a,fs,t),p(a,Wa,t),s(Wa,ol),p(a,gs,t),f(ba,a,t),p(a,_s,t),p(a,V,t),s(V,il),s(V,za),s(za,pl),s(V,cl),p(a,vs,t),p(a,S,t),s(S,G),s(G,Ce),f($a,Ce,null),s(S,hl),s(S,Me),s(Me,dl),p(a,js,t),p(a,K,t),s(K,ul),s(K,wa),s(wa,ml),s(K,fl),p(a,ys,t),f(xa,a,t),p(a,bs,t),p(a,H,t),s(H,Q),s(Q,ke),f(Ea,ke,null),s(H,gl),s(H,Pe),s(Pe,_l),p(a,$s,t),p(a,b,t),s(b,vl),s(b,Te),s(Te,jl),s(b,yl),s(b,De),s(De,bl),s(b,$l),s(b,Va),s(Va,wl),s(b,xl),p(a,ws,t),p(a,Ga,t),s(Ga,El),p(a,xs,t),f(Ca,a,t),p(a,Es,t),p(a,Ka,t),s(Ka,Cl),p(a,Cs,t),f(Ma,a,t),p(a,Ms,t),p(a,Qa,t),s(Qa,Ml),p(a,ks,t),f(ka,a,t),p(a,Ps,t),p(a,X,t),s(X,kl),s(X,Pa),s(Pa,Pl),s(X,Tl),p(a,Ts,t),p(a,M,t),s(M,Dl),s(M,Ae),s(Ae,Al),s(M,ql),s(M,qe),s(qe,Rl),s(M,Sl),p(a,Ds,t),f(Ta,a,t),p(a,As,t),p(a,Z,t),s(Z,Hl),s(Z,Da),s(Da,Il),s(Z,Nl),p(a,qs,t),p(a,I,t),s(I,aa),s(aa,Re),f(Aa,Re,null),s(I,Ll),s(I,Se),s(Se,Ol),p(a,Rs,t),p(a,y,t),s(y,Fl),s(y,He),s(He,Bl),s(y,Ul),s(y,Xa),s(Xa,Jl),s(y,Yl),s(y,Ie),s(Ie,Wl),s(y,zl),s(y,Ne),s(Ne,Vl),s(y,Gl),s(y,Le),s(Le,Kl),s(y,Ql),p(a,Ss,t),p(a,k,t),s(k,Xl),s(k,Oe),s(Oe,Zl),s(k,an),s(k,Za),s(Za,en),s(k,sn),p(a,Hs,t),f(qa,a,t),p(a,Is,t),p(a,ea,t),s(ea,tn),s(ea,Fe),s(Fe,ln),s(ea,nn),p(a,Ns,t),f(Ra,a,t),p(a,Ls,t),p(a,sa,t),s(sa,rn),s(sa,Be),s(Be,on),s(sa,pn),p(a,Os,t),f(Sa,a,t),p(a,Fs,t),p(a,ta,t),s(ta,cn),s(ta,Ue),s(Ue,hn),s(ta,dn),p(a,Bs,t),f(Ha,a,t),Us=!0},p:Ar,i(a){Us||(g(ra.$$.fragment,a),g(ia.$$.fragment,a),g(ca.$$.fragment,a),g(ha.$$.fragment,a),g(da.$$.fragment,a),g(ua.$$.fragment,a),g(fa.$$.fragment,a),g(ga.$$.fragment,a),g(_a.$$.fragment,a),g(va.$$.fragment,a),g(ja.$$.fragment,a),g(ya.$$.fragment,a),g(ba.$$.fragment,a),g($a.$$.fragment,a),g(xa.$$.fragment,a),g(Ea.$$.fragment,a),g(Ca.$$.fragment,a),g(Ma.$$.fragment,a),g(ka.$$.fragment,a),g(Ta.$$.fragment,a),g(Aa.$$.fragment,a),g(qa.$$.fragment,a),g(Ra.$$.fragment,a),g(Sa.$$.fragment,a),g(Ha.$$.fragment,a),Us=!0)},o(a){_(ra.$$.fragment,a),_(ia.$$.fragment,a),_(ca.$$.fragment,a),_(ha.$$.fragment,a),_(da.$$.fragment,a),_(ua.$$.fragment,a),_(fa.$$.fragment,a),_(ga.$$.fragment,a),_(_a.$$.fragment,a),_(va.$$.fragment,a),_(ja.$$.fragment,a),_(ya.$$.fragment,a),_(ba.$$.fragment,a),_($a.$$.fragment,a),_(xa.$$.fragment,a),_(Ea.$$.fragment,a),_(Ca.$$.fragment,a),_(Ma.$$.fragment,a),_(ka.$$.fragment,a),_(Ta.$$.fragment,a),_(Aa.$$.fragment,a),_(qa.$$.fragment,a),_(Ra.$$.fragment,a),_(Sa.$$.fragment,a),_(Ha.$$.fragment,a),Us=!1},d(a){e(P),a&&e(We),a&&e(T),v(ra),a&&e(ze),a&&e(w),a&&e(Ve),a&&e(D),v(ia),a&&e(Ge),a&&e(x),a&&e(Ke),v(ca,a),a&&e(Qe),a&&e(La),a&&e(Xe),a&&e(E),a&&e(Ze),a&&e(A),v(ha),a&&e(as),a&&e(q),v(da),a&&e(es),a&&e(J),a&&e(ss),v(ua,a),a&&e(ts),a&&e(Fa),a&&e(ls),a&&e(Y),a&&e(ns),v(fa,a),a&&e(rs),a&&e(Ua),a&&e(os),v(ga,a),a&&e(is),a&&e(R),v(_a),a&&e(ps),a&&e(z),a&&e(cs),v(va,a),a&&e(hs),a&&e(Ja),a&&e(ds),v(ja,a),a&&e(us),a&&e(Ya),a&&e(ms),v(ya,a),a&&e(fs),a&&e(Wa),a&&e(gs),v(ba,a),a&&e(_s),a&&e(V),a&&e(vs),a&&e(S),v($a),a&&e(js),a&&e(K),a&&e(ys),v(xa,a),a&&e(bs),a&&e(H),v(Ea),a&&e($s),a&&e(b),a&&e(ws),a&&e(Ga),a&&e(xs),v(Ca,a),a&&e(Es),a&&e(Ka),a&&e(Cs),v(Ma,a),a&&e(Ms),a&&e(Qa),a&&e(ks),v(ka,a),a&&e(Ps),a&&e(X),a&&e(Ts),a&&e(M),a&&e(Ds),v(Ta,a),a&&e(As),a&&e(Z),a&&e(qs),a&&e(I),v(Aa),a&&e(Rs),a&&e(y),a&&e(Ss),a&&e(k),a&&e(Hs),v(qa,a),a&&e(Is),a&&e(ea),a&&e(Ns),v(Ra,a),a&&e(Ls),a&&e(sa),a&&e(Os),v(Sa,a),a&&e(Fs),a&&e(ta),a&&e(Bs),v(Ha,a)}}}const Sr={local:"creating-and-sharing-model-cards",sections:[{local:"loading-a-model-card-from-the-hub",title:"Loading a Model Card from the Hub"},{local:"creating-model-cards",sections:[{local:"from-text",title:"From Text"},{local:"from-a-jinja-template",title:"From a Jinja Template"},{local:"from-the-default-template",title:"From the Default Template"}],title:"Creating Model Cards"},{local:"sharing-model-cards",sections:[{local:"including-evaluation-results",title:"Including Evaluation Results"}],title:"Sharing Model Cards"}],title:"Creating and Sharing Model Cards"};function Hr(un){return qr(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Or extends kr{constructor(P){super();Pr(this,P,Hr,Rr,Tr,{})}}export{Or as default,Sr as metadata};
