import{S as _r,i as br,s as xr,e as n,k as c,w as v,t as g,M as vr,c as r,d as s,m as p,a as o,x as $,h as m,b as h,G as a,g as _,y,q as j,o as D,B as C,v as $r,L as ce}from"../../chunks/vendor-hf-doc-builder.js";import{T as yr}from"../../chunks/Tip-hf-doc-builder.js";import{D as E}from"../../chunks/Docstring-hf-doc-builder.js";import{C as pe}from"../../chunks/CodeBlock-hf-doc-builder.js";import{I as mt}from"../../chunks/IconCopyLink-hf-doc-builder.js";import{E as ie}from"../../chunks/ExampleCodeBlock-hf-doc-builder.js";function jr(w){let i,x,f,l,b;return l=new pe({props:{code:`from huggingface_hub.repocard import RepoCard
card = RepoCard.load("nateraw/food")
assert card.data.tags == ["generated_from_trainer", "image-classification", "pytorch"]
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub.repocard <span class="hljs-keyword">import</span> RepoCard
<span class="hljs-meta">&gt;&gt;&gt; </span>card = RepoCard.load(<span class="hljs-string">&quot;nateraw/food&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">assert</span> card.data.tags == [<span class="hljs-string">&quot;generated_from_trainer&quot;</span>, <span class="hljs-string">&quot;image-classification&quot;</span>, <span class="hljs-string">&quot;pytorch&quot;</span>]
`}}),{c(){i=n("p"),x=g("Example:"),f=c(),v(l.$$.fragment)},l(t){i=r(t,"P",{});var d=o(i);x=m(d,"Example:"),d.forEach(s),f=p(t),$(l.$$.fragment,t)},m(t,d){_(t,i,d),a(i,x),_(t,f,d),y(l,t,d),b=!0},p:ce,i(t){b||(j(l.$$.fragment,t),b=!0)},o(t){D(l.$$.fragment,t),b=!1},d(t){t&&s(i),t&&s(f),C(l,t)}}}function Dr(w){let i,x,f,l,b;return l=new pe({props:{code:`from huggingface_hub.repocard import RepoCard
card = RepoCard("---\\nlanguage: en\\n---\\n# This is a test repo card")
card.save("/tmp/test.md")
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub.repocard <span class="hljs-keyword">import</span> RepoCard
<span class="hljs-meta">&gt;&gt;&gt; </span>card = RepoCard(<span class="hljs-string">&quot;---\\nlanguage: en\\n---\\n# This is a test repo card&quot;</span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>card.save(<span class="hljs-string">&quot;/tmp/test.md&quot;</span>)
`}}),{c(){i=n("p"),x=g("Example:"),f=c(),v(l.$$.fragment)},l(t){i=r(t,"P",{});var d=o(i);x=m(d,"Example:"),d.forEach(s),f=p(t),$(l.$$.fragment,t)},m(t,d){_(t,i,d),a(i,x),_(t,f,d),y(l,t,d),b=!0},p:ce,i(t){b||(j(l.$$.fragment,t),b=!0)},o(t){D(l.$$.fragment,t),b=!1},d(t){t&&s(i),t&&s(f),C(l,t)}}}function Cr(w){let i,x,f,l,b,t,d,ee,W,T,R,Me,U;return{c(){i=g(`Raises the following errors:
`),x=n("ul"),f=n("li"),l=n("a"),b=n("code"),t=g("RuntimeError"),d=g(`
if the card fails validation checks.`),ee=c(),W=n("li"),T=n("a"),R=n("code"),Me=g("HTTPError"),U=g(`
if the request to the Hub API fails for any other reason.`),this.h()},l(A){i=m(A,`Raises the following errors:
`),x=r(A,"UL",{});var F=o(x);f=r(F,"LI",{});var Y=o(f);l=r(Y,"A",{href:!0,rel:!0});var ha=o(l);b=r(ha,"CODE",{});var ua=o(b);t=m(ua,"RuntimeError"),ua.forEach(s),ha.forEach(s),d=m(Y,`
if the card fails validation checks.`),Y.forEach(s),ee=p(F),W=r(F,"LI",{});var de=o(W);T=r(de,"A",{href:!0,rel:!0});var M=o(T);R=r(M,"CODE",{});var O=o(R);Me=m(O,"HTTPError"),O.forEach(s),M.forEach(s),U=m(de,`
if the request to the Hub API fails for any other reason.`),de.forEach(s),F.forEach(s),this.h()},h(){h(l,"href","https://docs.python.org/3/library/exceptions.html#RuntimeError"),h(l,"rel","nofollow"),h(T,"href","https://2.python-requests.org/en/master/api/#requests.HTTPError"),h(T,"rel","nofollow")},m(A,F){_(A,i,F),_(A,x,F),a(x,f),a(f,l),a(l,b),a(b,t),a(f,d),a(x,ee),a(x,W),a(W,T),a(T,R),a(R,Me),a(W,U)},d(A){A&&s(i),A&&s(x)}}}function wr(w){let i,x,f,l,b;return l=new pe({props:{code:`from huggingface_hub import ModelCard, ModelCardData, EvalResult

# Using the Default Template
card_data = ModelCardData(
    language='en',
    license='mit',
    library_name='timm',
    tags=['image-classification', 'resnet'],
    datasets='beans',
    metrics=['accuracy'],
)
card = ModelCard.from_template(
    card_data,
    model_description='This model does x + y...'
)

# Including Evaluation Results
card_data = ModelCardData(
    language='en',
    tags=['image-classification', 'resnet'],
    eval_results=[
        EvalResult(
            task_type='image-classification',
            dataset_type='beans',
            dataset_name='Beans',
            metric_type='accuracy',
            metric_value=0.9,
        ),
    ],
    model_name='my-cool-model',
)
card = ModelCard.from_template(card_data)

# Using a Custom Template
card_data = ModelCardData(
    language='en',
    tags=['image-classification', 'resnet']
)
card = ModelCard.from_template(
    card_data=card_data,
    template_path='./src/huggingface_hub/templates/modelcard_template.md',
    custom_template_var='custom value',  # will be replaced in template if it exists
)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> ModelCard, ModelCardData, EvalResult

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Using the Default Template</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>card_data = ModelCardData(
<span class="hljs-meta">... </span>    language=<span class="hljs-string">&#x27;en&#x27;</span>,
<span class="hljs-meta">... </span>    license=<span class="hljs-string">&#x27;mit&#x27;</span>,
<span class="hljs-meta">... </span>    library_name=<span class="hljs-string">&#x27;timm&#x27;</span>,
<span class="hljs-meta">... </span>    tags=[<span class="hljs-string">&#x27;image-classification&#x27;</span>, <span class="hljs-string">&#x27;resnet&#x27;</span>],
<span class="hljs-meta">... </span>    datasets=<span class="hljs-string">&#x27;beans&#x27;</span>,
<span class="hljs-meta">... </span>    metrics=[<span class="hljs-string">&#x27;accuracy&#x27;</span>],
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>card = ModelCard.from_template(
<span class="hljs-meta">... </span>    card_data,
<span class="hljs-meta">... </span>    model_description=<span class="hljs-string">&#x27;This model does x + y...&#x27;</span>
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Including Evaluation Results</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>card_data = ModelCardData(
<span class="hljs-meta">... </span>    language=<span class="hljs-string">&#x27;en&#x27;</span>,
<span class="hljs-meta">... </span>    tags=[<span class="hljs-string">&#x27;image-classification&#x27;</span>, <span class="hljs-string">&#x27;resnet&#x27;</span>],
<span class="hljs-meta">... </span>    eval_results=[
<span class="hljs-meta">... </span>        EvalResult(
<span class="hljs-meta">... </span>            task_type=<span class="hljs-string">&#x27;image-classification&#x27;</span>,
<span class="hljs-meta">... </span>            dataset_type=<span class="hljs-string">&#x27;beans&#x27;</span>,
<span class="hljs-meta">... </span>            dataset_name=<span class="hljs-string">&#x27;Beans&#x27;</span>,
<span class="hljs-meta">... </span>            metric_type=<span class="hljs-string">&#x27;accuracy&#x27;</span>,
<span class="hljs-meta">... </span>            metric_value=<span class="hljs-number">0.9</span>,
<span class="hljs-meta">... </span>        ),
<span class="hljs-meta">... </span>    ],
<span class="hljs-meta">... </span>    model_name=<span class="hljs-string">&#x27;my-cool-model&#x27;</span>,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>card = ModelCard.from_template(card_data)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Using a Custom Template</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>card_data = ModelCardData(
<span class="hljs-meta">... </span>    language=<span class="hljs-string">&#x27;en&#x27;</span>,
<span class="hljs-meta">... </span>    tags=[<span class="hljs-string">&#x27;image-classification&#x27;</span>, <span class="hljs-string">&#x27;resnet&#x27;</span>]
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>card = ModelCard.from_template(
<span class="hljs-meta">... </span>    card_data=card_data,
<span class="hljs-meta">... </span>    template_path=<span class="hljs-string">&#x27;./src/huggingface_hub/templates/modelcard_template.md&#x27;</span>,
<span class="hljs-meta">... </span>    custom_template_var=<span class="hljs-string">&#x27;custom value&#x27;</span>,  <span class="hljs-comment"># will be replaced in template if it exists</span>
<span class="hljs-meta">... </span>)
`}}),{c(){i=n("p"),x=g("Example:"),f=c(),v(l.$$.fragment)},l(t){i=r(t,"P",{});var d=o(i);x=m(d,"Example:"),d.forEach(s),f=p(t),$(l.$$.fragment,t)},m(t,d){_(t,i,d),a(i,x),_(t,f,d),y(l,t,d),b=!0},p:ce,i(t){b||(j(l.$$.fragment,t),b=!0)},o(t){D(l.$$.fragment,t),b=!1},d(t){t&&s(i),t&&s(f),C(l,t)}}}function Er(w){let i,x,f,l,b;return l=new pe({props:{code:`from huggingface_hub import ModelCardData
card_data = ModelCardData(
    language="en",
    license="mit",
    library_name="timm",
    tags=['image-classification', 'resnet'],
)
card_data.to_dict()
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> ModelCardData
<span class="hljs-meta">&gt;&gt;&gt; </span>card_data = ModelCardData(
<span class="hljs-meta">... </span>    language=<span class="hljs-string">&quot;en&quot;</span>,
<span class="hljs-meta">... </span>    license=<span class="hljs-string">&quot;mit&quot;</span>,
<span class="hljs-meta">... </span>    library_name=<span class="hljs-string">&quot;timm&quot;</span>,
<span class="hljs-meta">... </span>    tags=[<span class="hljs-string">&#x27;image-classification&#x27;</span>, <span class="hljs-string">&#x27;resnet&#x27;</span>],
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>card_data.to_dict()
{<span class="hljs-string">&#x27;language&#x27;</span>: <span class="hljs-string">&#x27;en&#x27;</span>, <span class="hljs-string">&#x27;license&#x27;</span>: <span class="hljs-string">&#x27;mit&#x27;</span>, <span class="hljs-string">&#x27;library_name&#x27;</span>: <span class="hljs-string">&#x27;timm&#x27;</span>, <span class="hljs-string">&#x27;tags&#x27;</span>: [<span class="hljs-string">&#x27;image-classification&#x27;</span>, <span class="hljs-string">&#x27;resnet&#x27;</span>]}
`}}),{c(){i=n("p"),x=g("Example:"),f=c(),v(l.$$.fragment)},l(t){i=r(t,"P",{});var d=o(i);x=m(d,"Example:"),d.forEach(s),f=p(t),$(l.$$.fragment,t)},m(t,d){_(t,i,d),a(i,x),_(t,f,d),y(l,t,d),b=!0},p:ce,i(t){b||(j(l.$$.fragment,t),b=!0)},o(t){D(l.$$.fragment,t),b=!1},d(t){t&&s(i),t&&s(f),C(l,t)}}}function kr(w){let i,x,f,l,b;return l=new pe({props:{code:`from huggingface_hub import DatasetCard, DatasetCardData

# Using the Default Template
card_data = DatasetCardData(
    language='en',
    license='mit',
    annotations_creators='crowdsourced',
    task_categories=['text-classification'],
    task_ids=['sentiment-classification', 'text-scoring'],
    multilinguality='monolingual',
    pretty_name='My Text Classification Dataset',
)
card = DatasetCard.from_template(
    card_data,
    pretty_name=card_data.pretty_name,
)

# Using a Custom Template
card_data = DatasetCardData(
    language='en',
    license='mit',
)
card = DatasetCard.from_template(
    card_data=card_data,
    template_path='./src/huggingface_hub/templates/datasetcard_template.md',
    custom_template_var='custom value',  # will be replaced in template if it exists
)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> DatasetCard, DatasetCardData

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Using the Default Template</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>card_data = DatasetCardData(
<span class="hljs-meta">... </span>    language=<span class="hljs-string">&#x27;en&#x27;</span>,
<span class="hljs-meta">... </span>    license=<span class="hljs-string">&#x27;mit&#x27;</span>,
<span class="hljs-meta">... </span>    annotations_creators=<span class="hljs-string">&#x27;crowdsourced&#x27;</span>,
<span class="hljs-meta">... </span>    task_categories=[<span class="hljs-string">&#x27;text-classification&#x27;</span>],
<span class="hljs-meta">... </span>    task_ids=[<span class="hljs-string">&#x27;sentiment-classification&#x27;</span>, <span class="hljs-string">&#x27;text-scoring&#x27;</span>],
<span class="hljs-meta">... </span>    multilinguality=<span class="hljs-string">&#x27;monolingual&#x27;</span>,
<span class="hljs-meta">... </span>    pretty_name=<span class="hljs-string">&#x27;My Text Classification Dataset&#x27;</span>,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>card = DatasetCard.from_template(
<span class="hljs-meta">... </span>    card_data,
<span class="hljs-meta">... </span>    pretty_name=card_data.pretty_name,
<span class="hljs-meta">... </span>)

<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Using a Custom Template</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>card_data = DatasetCardData(
<span class="hljs-meta">... </span>    language=<span class="hljs-string">&#x27;en&#x27;</span>,
<span class="hljs-meta">... </span>    license=<span class="hljs-string">&#x27;mit&#x27;</span>,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>card = DatasetCard.from_template(
<span class="hljs-meta">... </span>    card_data=card_data,
<span class="hljs-meta">... </span>    template_path=<span class="hljs-string">&#x27;./src/huggingface_hub/templates/datasetcard_template.md&#x27;</span>,
<span class="hljs-meta">... </span>    custom_template_var=<span class="hljs-string">&#x27;custom value&#x27;</span>,  <span class="hljs-comment"># will be replaced in template if it exists</span>
<span class="hljs-meta">... </span>)
`}}),{c(){i=n("p"),x=g("Example:"),f=c(),v(l.$$.fragment)},l(t){i=r(t,"P",{});var d=o(i);x=m(d,"Example:"),d.forEach(s),f=p(t),$(l.$$.fragment,t)},m(t,d){_(t,i,d),a(i,x),_(t,f,d),y(l,t,d),b=!0},p:ce,i(t){b||(j(l.$$.fragment,t),b=!0)},o(t){D(l.$$.fragment,t),b=!1},d(t){t&&s(i),t&&s(f),C(l,t)}}}function Rr(w){let i,x,f,l,b;return l=new pe({props:{code:`from huggingface_hub.repocard_data import model_index_to_eval_results
# Define a minimal model index
model_index = [
    {
        "name": "my-cool-model",
        "results": [
            {
                "task": {
                    "type": "image-classification"
                },
                "dataset": {
                    "type": "beans",
                    "name": "Beans"
                },
                "metrics": [
                    {
                        "type": "accuracy",
                        "value": 0.9
                    }
                ]
            }
        ]
    }
]
model_name, eval_results = model_index_to_eval_results(model_index)
model_name
eval_results[0].task_type
eval_results[0].metric_type
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub.repocard_data <span class="hljs-keyword">import</span> model_index_to_eval_results
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Define a minimal model index</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>model_index = [
<span class="hljs-meta">... </span>    {
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;name&quot;</span>: <span class="hljs-string">&quot;my-cool-model&quot;</span>,
<span class="hljs-meta">... </span>        <span class="hljs-string">&quot;results&quot;</span>: [
<span class="hljs-meta">... </span>            {
<span class="hljs-meta">... </span>                <span class="hljs-string">&quot;task&quot;</span>: {
<span class="hljs-meta">... </span>                    <span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;image-classification&quot;</span>
<span class="hljs-meta">... </span>                },
<span class="hljs-meta">... </span>                <span class="hljs-string">&quot;dataset&quot;</span>: {
<span class="hljs-meta">... </span>                    <span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;beans&quot;</span>,
<span class="hljs-meta">... </span>                    <span class="hljs-string">&quot;name&quot;</span>: <span class="hljs-string">&quot;Beans&quot;</span>
<span class="hljs-meta">... </span>                },
<span class="hljs-meta">... </span>                <span class="hljs-string">&quot;metrics&quot;</span>: [
<span class="hljs-meta">... </span>                    {
<span class="hljs-meta">... </span>                        <span class="hljs-string">&quot;type&quot;</span>: <span class="hljs-string">&quot;accuracy&quot;</span>,
<span class="hljs-meta">... </span>                        <span class="hljs-string">&quot;value&quot;</span>: <span class="hljs-number">0.9</span>
<span class="hljs-meta">... </span>                    }
<span class="hljs-meta">... </span>                ]
<span class="hljs-meta">... </span>            }
<span class="hljs-meta">... </span>        ]
<span class="hljs-meta">... </span>    }
<span class="hljs-meta">... </span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>model_name, eval_results = model_index_to_eval_results(model_index)
<span class="hljs-meta">&gt;&gt;&gt; </span>model_name
<span class="hljs-string">&#x27;my-cool-model&#x27;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>eval_results[<span class="hljs-number">0</span>].task_type
<span class="hljs-string">&#x27;image-classification&#x27;</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>eval_results[<span class="hljs-number">0</span>].metric_type
<span class="hljs-string">&#x27;accuracy&#x27;</span>
`}}),{c(){i=n("p"),x=g("Example:"),f=c(),v(l.$$.fragment)},l(t){i=r(t,"P",{});var d=o(i);x=m(d,"Example:"),d.forEach(s),f=p(t),$(l.$$.fragment,t)},m(t,d){_(t,i,d),a(i,x),_(t,f,d),y(l,t,d),b=!0},p:ce,i(t){b||(j(l.$$.fragment,t),b=!0)},o(t){D(l.$$.fragment,t),b=!1},d(t){t&&s(i),t&&s(f),C(l,t)}}}function qr(w){let i,x,f,l,b;return l=new pe({props:{code:`from huggingface_hub.repocard_data import eval_results_to_model_index, EvalResult
# Define minimal eval_results
eval_results = [
    EvalResult(
        task_type="image-classification",  # Required
        dataset_type="beans",  # Required
        dataset_name="Beans",  # Required
        metric_type="accuracy",  # Required
        metric_value=0.9,  # Required
    )
]
eval_results_to_model_index("my-cool-model", eval_results)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub.repocard_data <span class="hljs-keyword">import</span> eval_results_to_model_index, EvalResult
<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-comment"># Define minimal eval_results</span>
<span class="hljs-meta">&gt;&gt;&gt; </span>eval_results = [
<span class="hljs-meta">... </span>    EvalResult(
<span class="hljs-meta">... </span>        task_type=<span class="hljs-string">&quot;image-classification&quot;</span>,  <span class="hljs-comment"># Required</span>
<span class="hljs-meta">... </span>        dataset_type=<span class="hljs-string">&quot;beans&quot;</span>,  <span class="hljs-comment"># Required</span>
<span class="hljs-meta">... </span>        dataset_name=<span class="hljs-string">&quot;Beans&quot;</span>,  <span class="hljs-comment"># Required</span>
<span class="hljs-meta">... </span>        metric_type=<span class="hljs-string">&quot;accuracy&quot;</span>,  <span class="hljs-comment"># Required</span>
<span class="hljs-meta">... </span>        metric_value=<span class="hljs-number">0.9</span>,  <span class="hljs-comment"># Required</span>
<span class="hljs-meta">... </span>    )
<span class="hljs-meta">... </span>]
<span class="hljs-meta">&gt;&gt;&gt; </span>eval_results_to_model_index(<span class="hljs-string">&quot;my-cool-model&quot;</span>, eval_results)
[{<span class="hljs-string">&#x27;name&#x27;</span>: <span class="hljs-string">&#x27;my-cool-model&#x27;</span>, <span class="hljs-string">&#x27;results&#x27;</span>: [{<span class="hljs-string">&#x27;task&#x27;</span>: {<span class="hljs-string">&#x27;type&#x27;</span>: <span class="hljs-string">&#x27;image-classification&#x27;</span>}, <span class="hljs-string">&#x27;dataset&#x27;</span>: {<span class="hljs-string">&#x27;name&#x27;</span>: <span class="hljs-string">&#x27;Beans&#x27;</span>, <span class="hljs-string">&#x27;type&#x27;</span>: <span class="hljs-string">&#x27;beans&#x27;</span>}, <span class="hljs-string">&#x27;metrics&#x27;</span>: [{<span class="hljs-string">&#x27;type&#x27;</span>: <span class="hljs-string">&#x27;accuracy&#x27;</span>, <span class="hljs-string">&#x27;value&#x27;</span>: <span class="hljs-number">0.9</span>}]}]}]
`}}),{c(){i=n("p"),x=g("Example:"),f=c(),v(l.$$.fragment)},l(t){i=r(t,"P",{});var d=o(i);x=m(d,"Example:"),d.forEach(s),f=p(t),$(l.$$.fragment,t)},m(t,d){_(t,i,d),a(i,x),_(t,f,d),y(l,t,d),b=!0},p:ce,i(t){b||(j(l.$$.fragment,t),b=!0)},o(t){D(l.$$.fragment,t),b=!1},d(t){t&&s(i),t&&s(f),C(l,t)}}}function Tr(w){let i,x,f,l,b;return l=new pe({props:{code:`from huggingface_hub import metadata_eval_result
results = metadata_eval_result(
        model_pretty_name="RoBERTa fine-tuned on ReactionGIF",
        task_pretty_name="Text Classification",
        task_id="text-classification",
        metrics_pretty_name="Accuracy",
        metrics_id="accuracy",
        metrics_value=0.2662102282047272,
        dataset_pretty_name="ReactionJPEG",
        dataset_id="julien-c/reactionjpeg",
        dataset_config="default",
        dataset_split="test",
)
results == {
    'model-index': [
        {
            'name': 'RoBERTa fine-tuned on ReactionGIF',
            'results': [
                {
                    'task': {
                        'type': 'text-classification',
                        'name': 'Text Classification'
                    },
                    'dataset': {
                        'name': 'ReactionJPEG',
                        'type': 'julien-c/reactionjpeg',
                        'config': 'default',
                        'split': 'test'
                    },
                    'metrics': [
                        {
                            'type': 'accuracy',
                            'value': 0.2662102282047272,
                            'name': 'Accuracy',
                            'verified': False
                        }
                    ]
                }
            ]
        }
    ]
}
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> metadata_eval_result
<span class="hljs-meta">&gt;&gt;&gt; </span>results = metadata_eval_result(
<span class="hljs-meta">... </span>        model_pretty_name=<span class="hljs-string">&quot;RoBERTa fine-tuned on ReactionGIF&quot;</span>,
<span class="hljs-meta">... </span>        task_pretty_name=<span class="hljs-string">&quot;Text Classification&quot;</span>,
<span class="hljs-meta">... </span>        task_id=<span class="hljs-string">&quot;text-classification&quot;</span>,
<span class="hljs-meta">... </span>        metrics_pretty_name=<span class="hljs-string">&quot;Accuracy&quot;</span>,
<span class="hljs-meta">... </span>        metrics_id=<span class="hljs-string">&quot;accuracy&quot;</span>,
<span class="hljs-meta">... </span>        metrics_value=<span class="hljs-number">0.2662102282047272</span>,
<span class="hljs-meta">... </span>        dataset_pretty_name=<span class="hljs-string">&quot;ReactionJPEG&quot;</span>,
<span class="hljs-meta">... </span>        dataset_id=<span class="hljs-string">&quot;julien-c/reactionjpeg&quot;</span>,
<span class="hljs-meta">... </span>        dataset_config=<span class="hljs-string">&quot;default&quot;</span>,
<span class="hljs-meta">... </span>        dataset_split=<span class="hljs-string">&quot;test&quot;</span>,
<span class="hljs-meta">... </span>)
<span class="hljs-meta">&gt;&gt;&gt; </span>results == {
<span class="hljs-meta">... </span>    <span class="hljs-string">&#x27;model-index&#x27;</span>: [
<span class="hljs-meta">... </span>        {
<span class="hljs-meta">... </span>            <span class="hljs-string">&#x27;name&#x27;</span>: <span class="hljs-string">&#x27;RoBERTa fine-tuned on ReactionGIF&#x27;</span>,
<span class="hljs-meta">... </span>            <span class="hljs-string">&#x27;results&#x27;</span>: [
<span class="hljs-meta">... </span>                {
<span class="hljs-meta">... </span>                    <span class="hljs-string">&#x27;task&#x27;</span>: {
<span class="hljs-meta">... </span>                        <span class="hljs-string">&#x27;type&#x27;</span>: <span class="hljs-string">&#x27;text-classification&#x27;</span>,
<span class="hljs-meta">... </span>                        <span class="hljs-string">&#x27;name&#x27;</span>: <span class="hljs-string">&#x27;Text Classification&#x27;</span>
<span class="hljs-meta">... </span>                    },
<span class="hljs-meta">... </span>                    <span class="hljs-string">&#x27;dataset&#x27;</span>: {
<span class="hljs-meta">... </span>                        <span class="hljs-string">&#x27;name&#x27;</span>: <span class="hljs-string">&#x27;ReactionJPEG&#x27;</span>,
<span class="hljs-meta">... </span>                        <span class="hljs-string">&#x27;type&#x27;</span>: <span class="hljs-string">&#x27;julien-c/reactionjpeg&#x27;</span>,
<span class="hljs-meta">... </span>                        <span class="hljs-string">&#x27;config&#x27;</span>: <span class="hljs-string">&#x27;default&#x27;</span>,
<span class="hljs-meta">... </span>                        <span class="hljs-string">&#x27;split&#x27;</span>: <span class="hljs-string">&#x27;test&#x27;</span>
<span class="hljs-meta">... </span>                    },
<span class="hljs-meta">... </span>                    <span class="hljs-string">&#x27;metrics&#x27;</span>: [
<span class="hljs-meta">... </span>                        {
<span class="hljs-meta">... </span>                            <span class="hljs-string">&#x27;type&#x27;</span>: <span class="hljs-string">&#x27;accuracy&#x27;</span>,
<span class="hljs-meta">... </span>                            <span class="hljs-string">&#x27;value&#x27;</span>: <span class="hljs-number">0.2662102282047272</span>,
<span class="hljs-meta">... </span>                            <span class="hljs-string">&#x27;name&#x27;</span>: <span class="hljs-string">&#x27;Accuracy&#x27;</span>,
<span class="hljs-meta">... </span>                            <span class="hljs-string">&#x27;verified&#x27;</span>: <span class="hljs-literal">False</span>
<span class="hljs-meta">... </span>                        }
<span class="hljs-meta">... </span>                    ]
<span class="hljs-meta">... </span>                }
<span class="hljs-meta">... </span>            ]
<span class="hljs-meta">... </span>        }
<span class="hljs-meta">... </span>    ]
<span class="hljs-meta">... </span>}
<span class="hljs-literal">True</span>
`}}),{c(){i=n("p"),x=g("Example:"),f=c(),v(l.$$.fragment)},l(t){i=r(t,"P",{});var d=o(i);x=m(d,"Example:"),d.forEach(s),f=p(t),$(l.$$.fragment,t)},m(t,d){_(t,i,d),a(i,x),_(t,f,d),y(l,t,d),b=!0},p:ce,i(t){b||(j(l.$$.fragment,t),b=!0)},o(t){D(l.$$.fragment,t),b=!1},d(t){t&&s(i),t&&s(f),C(l,t)}}}function Ar(w){let i,x,f,l,b;return l=new pe({props:{code:`from huggingface_hub import metadata_update
metadata = {'model-index': [{'name': 'RoBERTa fine-tuned on ReactionGIF',
            'results': [{'dataset': {'name': 'ReactionGIF',
                                     'type': 'julien-c/reactiongif'},
                          'metrics': [{'name': 'Recall',
                                       'type': 'recall',
                                       'value': 0.7762102282047272}],
                         'task': {'name': 'Text Classification',
                                  'type': 'text-classification'}}]}]}
url = metadata_update("julien-c/reactiongif-roberta", metadata)
`,highlighted:`<span class="hljs-meta">&gt;&gt;&gt; </span><span class="hljs-keyword">from</span> huggingface_hub <span class="hljs-keyword">import</span> metadata_update
<span class="hljs-meta">&gt;&gt;&gt; </span>metadata = {<span class="hljs-string">&#x27;model-index&#x27;</span>: [{<span class="hljs-string">&#x27;name&#x27;</span>: <span class="hljs-string">&#x27;RoBERTa fine-tuned on ReactionGIF&#x27;</span>,
<span class="hljs-meta">... </span>            <span class="hljs-string">&#x27;results&#x27;</span>: [{<span class="hljs-string">&#x27;dataset&#x27;</span>: {<span class="hljs-string">&#x27;name&#x27;</span>: <span class="hljs-string">&#x27;ReactionGIF&#x27;</span>,
<span class="hljs-meta">... </span>                                     <span class="hljs-string">&#x27;type&#x27;</span>: <span class="hljs-string">&#x27;julien-c/reactiongif&#x27;</span>},
<span class="hljs-meta">... </span>                          <span class="hljs-string">&#x27;metrics&#x27;</span>: [{<span class="hljs-string">&#x27;name&#x27;</span>: <span class="hljs-string">&#x27;Recall&#x27;</span>,
<span class="hljs-meta">... </span>                                       <span class="hljs-string">&#x27;type&#x27;</span>: <span class="hljs-string">&#x27;recall&#x27;</span>,
<span class="hljs-meta">... </span>                                       <span class="hljs-string">&#x27;value&#x27;</span>: <span class="hljs-number">0.7762102282047272</span>}],
<span class="hljs-meta">... </span>                         <span class="hljs-string">&#x27;task&#x27;</span>: {<span class="hljs-string">&#x27;name&#x27;</span>: <span class="hljs-string">&#x27;Text Classification&#x27;</span>,
<span class="hljs-meta">... </span>                                  <span class="hljs-string">&#x27;type&#x27;</span>: <span class="hljs-string">&#x27;text-classification&#x27;</span>}}]}]}
<span class="hljs-meta">&gt;&gt;&gt; </span>url = metadata_update(<span class="hljs-string">&quot;julien-c/reactiongif-roberta&quot;</span>, metadata)
`}}),{c(){i=n("p"),x=g("Example:"),f=c(),v(l.$$.fragment)},l(t){i=r(t,"P",{});var d=o(i);x=m(d,"Example:"),d.forEach(s),f=p(t),$(l.$$.fragment,t)},m(t,d){_(t,i,d),a(i,x),_(t,f,d),y(l,t,d),b=!0},p:ce,i(t){b||(j(l.$$.fragment,t),b=!0)},o(t){D(l.$$.fragment,t),b=!1},d(t){t&&s(i),t&&s(f),C(l,t)}}}function Mr(w){let i,x,f,l,b,t,d,ee,W,T,R,Me,U,A,F,Y,ha,ua,de,M,O,La,Ne,Bt,Ia,zt,ht,N,Gt,Pa,Jt,Wt,fa,Yt,Kt,_a,Qt,Xt,ut,k,Le,Zt,K,Ie,es,Ha,as,ts,Ua,ss,ns,Q,Pe,rs,Fa,os,ls,ge,is,me,He,cs,Oa,ps,ds,X,Ue,gs,Sa,ms,hs,he,us,Z,Fe,fs,Oe,_s,ba,bs,xs,vs,ue,ft,L,$s,xa,ys,js,va,Ds,Cs,$a,ws,Es,_t,S,Se,ks,fe,Ve,Rs,Va,qs,Ts,_e,Be,As,Ba,Ms,bt,ae,be,za,ze,Ns,Ga,Ls,xt,te,Ge,Is,I,Je,Ps,ya,Hs,We,Us,Fs,Ja,Os,Ss,xe,vt,V,Ye,Vs,Wa,Bs,zs,ve,$t,se,$e,Ya,Ke,Gs,Ka,Js,yt,ne,Qe,Ws,P,Xe,Ys,ja,Ks,Ze,Qs,Xs,Qa,Zs,en,ye,jt,re,ea,an,Xa,tn,Dt,oe,je,Za,aa,sn,et,nn,Ct,B,ta,rn,at,on,ln,sa,cn,na,pn,dn,wt,q,ra,gn,oa,mn,tt,hn,un,fn,Da,_n,la,bn,xn,De,Et,z,ia,vn,ca,$n,st,yn,jn,Dn,Ce,kt,G,pa,Cn,nt,wn,En,we,Rt,J,da,kn,rt,Rn,qn,Ee,qt;return t=new mt({}),Ne=new mt({}),Le=new E({props:{name:"class huggingface_hub.repocard.RepoCard",anchor:"huggingface_hub.repocard.RepoCard",parameters:[{name:"content",val:": str"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard.py#L42"}}),Ie=new E({props:{name:"from_template",anchor:"huggingface_hub.repocard.RepoCard.from_template",parameters:[{name:"card_data",val:": CardData"},{name:"template_path",val:": typing.Optional[str] = None"},{name:"**template_kwargs",val:""}],parametersDescription:[{anchor:"huggingface_hub.repocard.RepoCard.from_template.card_data",description:`<strong>card_data</strong> (<code>huggingface_hub.CardData</code>) &#x2014;
A huggingface_hub.CardData instance containing the metadata you want to include in the YAML
header of the repo card on the Hugging Face Hub.`,name:"card_data"},{anchor:"huggingface_hub.repocard.RepoCard.from_template.template_path",description:`<strong>template_path</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A path to a markdown file with optional Jinja template variables that can be filled
in with <code>template_kwargs</code>. Defaults to the default template.`,name:"template_path"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard.py#L271",returnDescription:`
<p>A RepoCard instance with the specified card data and content from the
template.</p>
`,returnType:`
<p><a
  href="/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.repocard.RepoCard"
>huggingface_hub.repocard.RepoCard</a></p>
`}}),Pe=new E({props:{name:"load",anchor:"huggingface_hub.repocard.RepoCard.load",parameters:[{name:"repo_id_or_path",val:": typing.Union[str, pathlib.Path]"},{name:"repo_type",val:" = None"},{name:"token",val:" = None"}],parametersDescription:[{anchor:"huggingface_hub.repocard.RepoCard.load.repo_id_or_path",description:`<strong>repo_id_or_path</strong> (<code>Union[str, Path]</code>) &#x2014;
The repo ID associated with a Hugging Face Hub repo or a local filepath.`,name:"repo_id_or_path"},{anchor:"huggingface_hub.repocard.RepoCard.load.repo_type",description:`<strong>repo_type</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The type of Hugging Face repo to push to. Defaults to None, which will use
use &#x201C;model&#x201D;. Other options are &#x201C;dataset&#x201D; and &#x201C;space&#x201D;. Not used when loading from
a local filepath. If this is called from a child class, the default value will be
the child class&#x2019;s <code>repo_type</code>.`,name:"repo_type"},{anchor:"huggingface_hub.repocard.RepoCard.load.token",description:`<strong>token</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Authentication token, obtained with <code>huggingface_hub.HfApi.login</code> method. Will default to
the stored token.`,name:"token"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard.py#L124",returnDescription:`
<p>The RepoCard (or subclass) initialized from the repo\u2019s
README.md file or filepath.</p>
`,returnType:`
<p><a
  href="/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.repocard.RepoCard"
>huggingface_hub.repocard.RepoCard</a></p>
`}}),ge=new ie({props:{anchor:"huggingface_hub.repocard.RepoCard.load.example",$$slots:{default:[jr]},$$scope:{ctx:w}}}),He=new E({props:{name:"push_to_hub",anchor:"huggingface_hub.repocard.RepoCard.push_to_hub",parameters:[{name:"repo_id",val:""},{name:"token",val:" = None"},{name:"repo_type",val:" = None"},{name:"commit_message",val:" = None"},{name:"commit_description",val:" = None"},{name:"revision",val:" = None"},{name:"create_pr",val:" = None"},{name:"parent_commit",val:" = None"}],parametersDescription:[{anchor:"huggingface_hub.repocard.RepoCard.push_to_hub.repo_id",description:`<strong>repo_id</strong> (<code>str</code>) &#x2014;
The repo ID of the Hugging Face Hub repo to push to. Example: &#x201C;nateraw/food&#x201D;.`,name:"repo_id"},{anchor:"huggingface_hub.repocard.RepoCard.push_to_hub.token",description:`<strong>token</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Authentication token, obtained with <code>huggingface_hub.HfApi.login</code> method. Will default to
the stored token.`,name:"token"},{anchor:"huggingface_hub.repocard.RepoCard.push_to_hub.repo_type",description:`<strong>repo_type</strong> (<code>str</code>, <em>optional</em>, defaults to &#x201C;model&#x201D;) &#x2014;
The type of Hugging Face repo to push to. Options are &#x201C;model&#x201D;, &#x201C;dataset&#x201D;, and &#x201C;space&#x201D;. If this
function is called by a child class, it will default to the child class&#x2019;s <code>repo_type</code>.`,name:"repo_type"},{anchor:"huggingface_hub.repocard.RepoCard.push_to_hub.commit_message",description:`<strong>commit_message</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The summary / title / first line of the generated commit.`,name:"commit_message"},{anchor:"huggingface_hub.repocard.RepoCard.push_to_hub.commit_description",description:`<strong>commit_description</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The description of the generated commit.`,name:"commit_description"},{anchor:"huggingface_hub.repocard.RepoCard.push_to_hub.revision",description:`<strong>revision</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The git revision to commit from. Defaults to the head of the <code>&quot;main&quot;</code> branch.`,name:"revision"},{anchor:"huggingface_hub.repocard.RepoCard.push_to_hub.create_pr",description:`<strong>create_pr</strong> (<code>bool</code>, <em>optional</em>) &#x2014;
Whether or not to create a Pull Request with this commit. Defaults to <code>False</code>.`,name:"create_pr"},{anchor:"huggingface_hub.repocard.RepoCard.push_to_hub.parent_commit",description:`<strong>parent_commit</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
If specified and <code>create_pr</code> is <code>False</code>, the commit will fail if <code>revision</code> does not point to <code>parent_commit</code>.
If specified and <code>create_pr</code> is <code>True</code>, the pull request will be created from <code>parent_commit</code>.
Specifying <code>parent_commit</code> ensures the repo has not changed before committing the changes, and can be
especially useful if the repo is updated / committed to concurrently.`,name:"parent_commit"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard.py#L208",returnDescription:`
<p>URL of the commit which updated the card metadata.</p>
`,returnType:`
<p><code>str</code></p>
`}}),Ue=new E({props:{name:"save",anchor:"huggingface_hub.repocard.RepoCard.save",parameters:[{name:"filepath",val:": typing.Union[pathlib.Path, str]"}],parametersDescription:[{anchor:"huggingface_hub.repocard.RepoCard.save.filepath",description:"<strong>filepath</strong> (<code>Union[Path, str]</code>) &#x2014; Filepath to the markdown file to save.",name:"filepath"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard.py#L106"}}),he=new ie({props:{anchor:"huggingface_hub.repocard.RepoCard.save.example",$$slots:{default:[Dr]},$$scope:{ctx:w}}}),Fe=new E({props:{name:"validate",anchor:"huggingface_hub.repocard.RepoCard.validate",parameters:[{name:"repo_type",val:" = None"}],parametersDescription:[{anchor:"huggingface_hub.repocard.RepoCard.validate.repo_type",description:`<strong>repo_type</strong> (<code>str</code>, <em>optional</em>, defaults to &#x201C;model&#x201D;) &#x2014;
The type of Hugging Face repo to push to. Options are &#x201C;model&#x201D;, &#x201C;dataset&#x201D;, and &#x201C;space&#x201D;.
If this function is called from a child class, the default will be the child class&#x2019;s <code>repo_type</code>.`,name:"repo_type"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard.py#L167"}}),ue=new yr({props:{$$slots:{default:[Cr]},$$scope:{ctx:w}}}),Se=new E({props:{name:"class huggingface_hub.CardData",anchor:"huggingface_hub.CardData",parameters:[{name:"**kwargs",val:""}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard_data.py#L121"}}),Ve=new E({props:{name:"to_dict",anchor:"huggingface_hub.CardData.to_dict",parameters:[],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard_data.py#L125",returnDescription:`
<p>CardData represented as a dictionary ready to be dumped to a YAML
block for inclusion in a README.md file.</p>
`,returnType:`
<p><code>dict</code></p>
`}}),Be=new E({props:{name:"to_yaml",anchor:"huggingface_hub.CardData.to_yaml",parameters:[{name:"line_break",val:" = None"}],parametersDescription:[{anchor:"huggingface_hub.CardData.to_yaml.line_break",description:`<strong>line_break</strong> (str, <em>optional</em>) &#x2014;
The line break to use when dumping to yaml.`,name:"line_break"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard_data.py#L145",returnDescription:`
<p>CardData represented as a YAML block.</p>
`,returnType:`
<p><code>str</code></p>
`}}),ze=new mt({}),Ge=new E({props:{name:"class huggingface_hub.ModelCard",anchor:"huggingface_hub.ModelCard",parameters:[{name:"content",val:": str"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard.py#L309"}}),Je=new E({props:{name:"from_template",anchor:"huggingface_hub.ModelCard.from_template",parameters:[{name:"card_data",val:": CardData"},{name:"template_path",val:": typing.Optional[str] = None"},{name:"**template_kwargs",val:""}],parametersDescription:[{anchor:"huggingface_hub.ModelCard.from_template.card_data",description:`<strong>card_data</strong> (<code>huggingface_hub.ModelCardData</code>) &#x2014;
A huggingface_hub.ModelCardData instance containing the metadata you want to include in the YAML
header of the model card on the Hugging Face Hub.`,name:"card_data"},{anchor:"huggingface_hub.ModelCard.from_template.template_path",description:`<strong>template_path</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A path to a markdown file with optional Jinja template variables that can be filled
in with <code>template_kwargs</code>. Defaults to the default template.`,name:"template_path"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard.py#L314",returnDescription:`
<p>A ModelCard instance with the specified card data and content from the
template.</p>
`,returnType:`
<p><a
  href="/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.ModelCard"
>huggingface_hub.ModelCard</a></p>
`}}),xe=new ie({props:{anchor:"huggingface_hub.ModelCard.from_template.example",$$slots:{default:[wr]},$$scope:{ctx:w}}}),Ye=new E({props:{name:"class huggingface_hub.ModelCardData",anchor:"huggingface_hub.ModelCardData",parameters:[{name:"language",val:": typing.Union[typing.List[str], str, NoneType] = None"},{name:"license",val:": typing.Optional[str] = None"},{name:"library_name",val:": typing.Optional[str] = None"},{name:"tags",val:": typing.Optional[typing.List[str]] = None"},{name:"datasets",val:": typing.Union[typing.List[str], str, NoneType] = None"},{name:"metrics",val:": typing.Union[typing.List[str], str, NoneType] = None"},{name:"eval_results",val:": typing.Optional[typing.List[huggingface_hub.repocard_data.EvalResult]] = None"},{name:"model_name",val:": typing.Optional[str] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"huggingface_hub.ModelCardData.language",description:`<strong>language</strong> (<code>Union[str, List[str]]</code>, <em>optional</em>) &#x2014;
Language of model&#x2019;s training data or metadata. It must be an ISO 639-1, 639-2 or
639-3 code (two/three letters), or a special value like &#x201C;code&#x201D;, &#x201C;multilingual&#x201D;. Defaults to <code>None</code>.`,name:"language"},{anchor:"huggingface_hub.ModelCardData.license",description:`<strong>license</strong> (<code>str</code>, <em>optional</em>) &#x2014;
License of this model. Example: apache-2.0 or any license from
<a href="https://huggingface.co/docs/hub/repositories-licenses" rel="nofollow">https://huggingface.co/docs/hub/repositories-licenses</a>. Defaults to None.`,name:"license"},{anchor:"huggingface_hub.ModelCardData.library_name",description:`<strong>library_name</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Name of library used by this model. Example: keras or any library from
<a href="https://github.com/huggingface/hub-docs/blob/main/js/src/lib/interfaces/Libraries.ts" rel="nofollow">https://github.com/huggingface/hub-docs/blob/main/js/src/lib/interfaces/Libraries.ts</a>.
Defaults to None.`,name:"library_name"},{anchor:"huggingface_hub.ModelCardData.tags",description:`<strong>tags</strong> (<code>List[str]</code>, <em>optional</em>) &#x2014;
List of tags to add to your model that can be used when filtering on the Hugging
Face Hub. Defaults to None.`,name:"tags"},{anchor:"huggingface_hub.ModelCardData.datasets",description:`<strong>datasets</strong> (<code>Union[str, List[str]]</code>, <em>optional</em>) &#x2014;
Dataset or list of datasets that were used to train this model. Should be a dataset ID
found on <a href="https://hf.co/datasets" rel="nofollow">https://hf.co/datasets</a>. Defaults to None.`,name:"datasets"},{anchor:"huggingface_hub.ModelCardData.metrics",description:`<strong>metrics</strong> (<code>Union[str, List[str]]</code>, <em>optional</em>) &#x2014;
List of metrics used to evaluate this model. Should be a metric name that can be found
at <a href="https://hf.co/metrics" rel="nofollow">https://hf.co/metrics</a>. Example: &#x2018;accuracy&#x2019;. Defaults to None.`,name:"metrics"},{anchor:"huggingface_hub.ModelCardData.eval_results",description:`<strong>eval_results</strong> (<code>Union[List[EvalResult], EvalResult]</code>, <em>optional</em>) &#x2014;
List of <code>huggingface_hub.EvalResult</code> that define evaluation results of the model. If provided,
<code>model_name</code> kwarg must be provided. Defaults to <code>None</code>.`,name:"eval_results"},{anchor:"huggingface_hub.ModelCardData.model_name",description:`<strong>model_name</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A name for this model. Required if you provide <code>eval_results</code>. It is used along with
<code>eval_results</code> to construct the <code>model-index</code> within the card&#x2019;s metadata. The name
you supply here is what will be used on PapersWithCode&#x2019;s leaderboards. Defaults to None.`,name:"model_name"},{anchor:"huggingface_hub.ModelCardData.kwargs",description:`<strong>kwargs</strong> (<code>dict</code>, <em>optional</em>) &#x2014;
Additional metadata that will be added to the model card. Defaults to None.`,name:"kwargs"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard_data.py#L161"}}),ve=new ie({props:{anchor:"huggingface_hub.ModelCardData.example",$$slots:{default:[Er]},$$scope:{ctx:w}}}),Ke=new mt({}),Qe=new E({props:{name:"class huggingface_hub.DatasetCard",anchor:"huggingface_hub.DatasetCard",parameters:[{name:"content",val:": str"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard.py#L389"}}),Xe=new E({props:{name:"from_template",anchor:"huggingface_hub.DatasetCard.from_template",parameters:[{name:"card_data",val:": CardData"},{name:"template_path",val:": typing.Optional[str] = None"},{name:"**template_kwargs",val:""}],parametersDescription:[{anchor:"huggingface_hub.DatasetCard.from_template.card_data",description:`<strong>card_data</strong> (<code>huggingface_hub.DatasetCardData</code>) &#x2014;
A huggingface_hub.DatasetCardData instance containing the metadata you want to include in the YAML
header of the dataset card on the Hugging Face Hub.`,name:"card_data"},{anchor:"huggingface_hub.DatasetCard.from_template.template_path",description:`<strong>template_path</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A path to a markdown file with optional Jinja template variables that can be filled
in with <code>template_kwargs</code>. Defaults to the default template.`,name:"template_path"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard.py#L394",returnDescription:`
<p>A DatasetCard instance with the specified card data and content from the
template.</p>
`,returnType:`
<p><a
  href="/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.DatasetCard"
>huggingface_hub.DatasetCard</a></p>
`}}),ye=new ie({props:{anchor:"huggingface_hub.DatasetCard.from_template.example",$$slots:{default:[kr]},$$scope:{ctx:w}}}),ea=new E({props:{name:"class huggingface_hub.DatasetCardData",anchor:"huggingface_hub.DatasetCardData",parameters:[{name:"language",val:": typing.Union[typing.List[str], str, NoneType] = None"},{name:"license",val:": typing.Union[typing.List[str], str, NoneType] = None"},{name:"annotations_creators",val:": typing.Union[typing.List[str], str, NoneType] = None"},{name:"language_creators",val:": typing.Union[typing.List[str], str, NoneType] = None"},{name:"multilinguality",val:": typing.Union[typing.List[str], str, NoneType] = None"},{name:"size_categories",val:": typing.Union[typing.List[str], str, NoneType] = None"},{name:"source_datasets",val:": typing.Union[typing.List[str], str, NoneType] = None"},{name:"task_categories",val:": typing.Union[typing.List[str], str, NoneType] = None"},{name:"task_ids",val:": typing.Union[typing.List[str], str, NoneType] = None"},{name:"paperswithcode_id",val:": typing.Optional[str] = None"},{name:"pretty_name",val:": typing.Optional[str] = None"},{name:"train_eval_index",val:": typing.Optional[typing.Dict] = None"},{name:"configs",val:": typing.Union[typing.List[str], str, NoneType] = None"},{name:"**kwargs",val:""}],parametersDescription:[{anchor:"huggingface_hub.DatasetCardData.language",description:`<strong>language</strong> (<code>Union[str, List[str]]</code>, <em>optional</em>) &#x2014;
Language of dataset&#x2019;s data or metadata. It must be an ISO 639-1, 639-2 or
639-3 code (two/three letters), or a special value like &#x201C;code&#x201D;, &#x201C;multilingual&#x201D;.`,name:"language"},{anchor:"huggingface_hub.DatasetCardData.license",description:`<strong>license</strong> (<code>Union[str, List[str]]</code>, <em>optional</em>) &#x2014;
License(s) of this dataset. Example: apache-2.0 or any license from
<a href="https://huggingface.co/docs/hub/repositories-licenses" rel="nofollow">https://huggingface.co/docs/hub/repositories-licenses</a>.`,name:"license"},{anchor:"huggingface_hub.DatasetCardData.annotations_creators",description:`<strong>annotations_creators</strong> (<code>Union[str, List[str]]</code>, <em>optional</em>) &#x2014;
How the annotations for the dataset were created.
Options are: &#x2018;found&#x2019;, &#x2018;crowdsourced&#x2019;, &#x2018;expert-generated&#x2019;, &#x2018;machine-generated&#x2019;, &#x2018;no-annotation&#x2019;, &#x2018;other&#x2019;.`,name:"annotations_creators"},{anchor:"huggingface_hub.DatasetCardData.language_creators",description:`<strong>language_creators</strong> (<code>Union[str, List[str]]</code>, <em>optional</em>) &#x2014;
How the text-based data in the dataset was created.
Options are: &#x2018;found&#x2019;, &#x2018;crowdsourced&#x2019;, &#x2018;expert-generated&#x2019;, &#x2018;machine-generated&#x2019;, &#x2018;other&#x2019;`,name:"language_creators"},{anchor:"huggingface_hub.DatasetCardData.multilinguality",description:`<strong>multilinguality</strong> (<code>Union[str, List[str]]</code>, <em>optional</em>) &#x2014;
Whether the dataset is multilingual.
Options are: &#x2018;monolingual&#x2019;, &#x2018;multilingual&#x2019;, &#x2018;translation&#x2019;, &#x2018;other&#x2019;.`,name:"multilinguality"},{anchor:"huggingface_hub.DatasetCardData.size_categories",description:`<strong>size_categories</strong> (<code>Union[str, List[str]]</code>, <em>optional</em>) &#x2014;
The number of examples in the dataset. Options are: &#x2018;n&lt;1K&#x2019;, &#x2018;1K<n<10k\u2019, \u201810k<n<100k\u2019, \u2018100k<n<1m\u2019, \u20181m<n<10m\u2019, \u201810m<n<100m\u2019, \u2018100m<n<1b\u2019, \u20181b<n<10b\u2019, \u201810b<n<100b\u2019, \u2018100b<n<1t\u2019, \u2018n>1T&#x2019;, and &#x2018;other&#x2019;.</n<10k\u2019,>`,name:"size_categories"},{anchor:"huggingface_hub.DatasetCardData.source_datasets",description:`<strong>source_datasets</strong> (<code>Union[str, List[str]]</code>, <em>optional</em>) &#x2014;
Indicates whether the dataset is an original dataset or extended from another existing dataset.
Options are: &#x2018;original&#x2019; and &#x2018;extended&#x2019;.`,name:"source_datasets"},{anchor:"huggingface_hub.DatasetCardData.task_categories",description:`<strong>task_categories</strong> (<code>Union[str, List[str]]</code>, <em>optional</em>) &#x2014;
What categories of task does the dataset support?`,name:"task_categories"},{anchor:"huggingface_hub.DatasetCardData.task_ids",description:`<strong>task_ids</strong> (<code>Union[str, List[str]]</code>, <em>optional</em>) &#x2014;
What specific tasks does the dataset support?`,name:"task_ids"},{anchor:"huggingface_hub.DatasetCardData.paperswithcode_id",description:`<strong>paperswithcode_id</strong> (<code>str</code>, <em>optional</em>) &#x2014;
ID of the dataset on PapersWithCode.`,name:"paperswithcode_id"},{anchor:"huggingface_hub.DatasetCardData.pretty_name",description:`<strong>pretty_name</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A more human-readable name for the dataset. (ex. &#x201C;Cats vs. Dogs&#x201D;)`,name:"pretty_name"},{anchor:"huggingface_hub.DatasetCardData.train_eval_index",description:`<strong>train_eval_index</strong> (<code>Dict</code>, <em>optional</em>) &#x2014;
A dictionary that describes the necessary spec for doing evaluation on the Hub.
If not provided, it will be gathered from the &#x2018;train-eval-index&#x2019; key of the kwargs.`,name:"train_eval_index"},{anchor:"huggingface_hub.DatasetCardData.configs",description:`<strong>configs</strong> (<code>Union[str, List[str]]</code>, <em>optional</em>) &#x2014;
A list of the available dataset configs for the dataset.`,name:"configs"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard_data.py#L260"}}),aa=new mt({}),ta=new E({props:{name:"class huggingface_hub.EvalResult",anchor:"huggingface_hub.EvalResult",parameters:[{name:"task_type",val:": str"},{name:"dataset_type",val:": str"},{name:"dataset_name",val:": str"},{name:"metric_type",val:": str"},{name:"metric_value",val:": typing.Any"},{name:"task_name",val:": typing.Optional[str] = None"},{name:"dataset_config",val:": typing.Optional[str] = None"},{name:"dataset_split",val:": typing.Optional[str] = None"},{name:"dataset_revision",val:": typing.Optional[str] = None"},{name:"dataset_args",val:": typing.Union[typing.Dict[str, typing.Any], NoneType] = None"},{name:"metric_name",val:": typing.Optional[str] = None"},{name:"metric_config",val:": typing.Optional[str] = None"},{name:"metric_args",val:": typing.Union[typing.Dict[str, typing.Any], NoneType] = None"},{name:"verified",val:": typing.Optional[bool] = None"}],parametersDescription:[{anchor:"huggingface_hub.EvalResult.task_type",description:`<strong>task_type</strong> (<code>str</code>) &#x2014;
The task identifier. Example: &#x201C;image-classification&#x201D;.`,name:"task_type"},{anchor:"huggingface_hub.EvalResult.dataset_type",description:`<strong>dataset_type</strong> (<code>str</code>) &#x2014;
The dataset identifier. Example: common_voice. Use dataset id from <a href="https://hf.co/datasets" rel="nofollow">https://hf.co/datasets</a>.`,name:"dataset_type"},{anchor:"huggingface_hub.EvalResult.dataset_name",description:`<strong>dataset_name</strong> (<code>str</code>) &#x2014;
A pretty name for the dataset. Example: &#x201C;Common Voice (French)&#x201C;.`,name:"dataset_name"},{anchor:"huggingface_hub.EvalResult.metric_type",description:`<strong>metric_type</strong> (<code>str</code>) &#x2014;
The metric identifier. Example: &#x201C;wer&#x201D;. Use metric id from <a href="https://hf.co/metrics" rel="nofollow">https://hf.co/metrics</a>.`,name:"metric_type"},{anchor:"huggingface_hub.EvalResult.metric_value",description:`<strong>metric_value</strong> (<code>Any</code>) &#x2014;
The metric value. Example: 0.9 or &#x201C;20.0 &#xB1; 1.2&#x201D;.`,name:"metric_value"},{anchor:"huggingface_hub.EvalResult.task_name",description:`<strong>task_name</strong> (<code>str</code>, <em>optional</em>) &#x2014;
A pretty name for the task. Example: &#x201C;Speech Recognition&#x201D;.`,name:"task_name"},{anchor:"huggingface_hub.EvalResult.dataset_config",description:`<strong>dataset_config</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The name of the dataset configuration used in <code>load_dataset()</code>.
Example: fr in <code>load_dataset(&quot;common_voice&quot;, &quot;fr&quot;)</code>. See the <code>datasets</code> docs for more info:
<a href="https://hf.co/docs/datasets/package_reference/loading_methods#datasets.load_dataset.name" rel="nofollow">https://hf.co/docs/datasets/package_reference/loading_methods#datasets.load_dataset.name</a>`,name:"dataset_config"},{anchor:"huggingface_hub.EvalResult.dataset_split",description:`<strong>dataset_split</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The split used in <code>load_dataset()</code>. Example: &#x201C;test&#x201D;.`,name:"dataset_split"},{anchor:"huggingface_hub.EvalResult.dataset_revision",description:`<strong>dataset_revision</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The revision (AKA Git Sha) of the dataset used in <code>load_dataset()</code>.
Example: 5503434ddd753f426f4b38109466949a1217c2bb`,name:"dataset_revision"},{anchor:"huggingface_hub.EvalResult.dataset_args",description:`<strong>dataset_args</strong> (<code>Dict[str, Any]</code>, <em>optional</em>) &#x2014;
The arguments passed during <code>Metric.compute()</code>. Example for <code>bleu</code>: max_order: 4.
metric_name &#x2014; (<code>str</code>, <em>optional</em>):
A pretty name for the metric. Example: &#x201C;Test WER&#x201D;.
metric_config &#x2014; (<code>str</code>, <em>optional</em>):
The name of the metric configuration used in <code>load_metric()</code>.
Example: bleurt-large-512 in <code>load_metric(&quot;bleurt&quot;, &quot;bleurt-large-512&quot;)</code>.
See the <code>datasets</code> docs for more info: <a href="https://huggingface.co/docs/datasets/v2.1.0/en/loading#load-configurations" rel="nofollow">https://huggingface.co/docs/datasets/v2.1.0/en/loading#load-configurations</a>
metric_args &#x2014; (<code>Dict[str, Any]</code>, <em>optional</em>):
The arguments passed during <code>Metric.compute()</code>. Example for <code>bleu</code>: max_order: 4
verified &#x2014; (<code>bool</code>, <em>optional</em>):
If true, indicates that evaluation was generated by Hugging Face (vs. self-reported).`,name:"dataset_args"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard_data.py#L15"}}),ra=new E({props:{name:"huggingface_hub.repocard_data.model_index_to_eval_results",anchor:"huggingface_hub.repocard_data.model_index_to_eval_results",parameters:[{name:"model_index",val:": typing.List[typing.Dict[str, typing.Any]]"}],parametersDescription:[{anchor:"huggingface_hub.repocard_data.model_index_to_eval_results.model_index",description:`<strong>model_index</strong> (<code>List[Dict[str, Any]]</code>) &#x2014;
A model index data structure, likely coming from a README.md file on the
Hugging Face Hub.`,name:"model_index"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard_data.py#L338",returnDescription:`
<p>The name of the model as found in the model index. This is used as the
identifier for the model on leaderboards like PapersWithCode.
eval_results (<code>List[EvalResult]</code>):
A list of <code>huggingface_hub.EvalResult</code> objects containing the metrics
reported in the provided model_index.</p>
`,returnType:`
<p>model_name (<code>str</code>)</p>
`}}),De=new ie({props:{anchor:"huggingface_hub.repocard_data.model_index_to_eval_results.example",$$slots:{default:[Rr]},$$scope:{ctx:w}}}),ia=new E({props:{name:"huggingface_hub.repocard_data.eval_results_to_model_index",anchor:"huggingface_hub.repocard_data.eval_results_to_model_index",parameters:[{name:"model_name",val:": str"},{name:"eval_results",val:": typing.List[huggingface_hub.repocard_data.EvalResult]"}],parametersDescription:[{anchor:"huggingface_hub.repocard_data.eval_results_to_model_index.model_name",description:`<strong>model_name</strong> (<code>str</code>) &#x2014;
Name of the model (ex. &#x201C;my-cool-model&#x201D;). This is used as the identifier
for the model on leaderboards like PapersWithCode.`,name:"model_name"},{anchor:"huggingface_hub.repocard_data.eval_results_to_model_index.eval_results",description:`<strong>eval_results</strong> (<code>List[EvalResult]</code>) &#x2014;
List of <code>huggingface_hub.EvalResult</code> objects containing the metrics to be
reported in the model-index.`,name:"eval_results"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard_data.py#L454",returnDescription:`
<p>The eval_results converted to a model-index.</p>
`,returnType:`
<p>model_index (<code>List[Dict[str, Any]]</code>)</p>
`}}),Ce=new ie({props:{anchor:"huggingface_hub.repocard_data.eval_results_to_model_index.example",$$slots:{default:[qr]},$$scope:{ctx:w}}}),pa=new E({props:{name:"huggingface_hub.metadata_eval_result",anchor:"huggingface_hub.metadata_eval_result",parameters:[{name:"model_pretty_name",val:": str"},{name:"task_pretty_name",val:": str"},{name:"task_id",val:": str"},{name:"metrics_pretty_name",val:": str"},{name:"metrics_id",val:": str"},{name:"metrics_value",val:": typing.Any"},{name:"dataset_pretty_name",val:": str"},{name:"dataset_id",val:": str"},{name:"metrics_config",val:": typing.Optional[str] = None"},{name:"metrics_verified",val:": typing.Optional[bool] = False"},{name:"dataset_config",val:": typing.Optional[str] = None"},{name:"dataset_split",val:": typing.Optional[str] = None"},{name:"dataset_revision",val:": typing.Optional[str] = None"}],parametersDescription:[{anchor:"huggingface_hub.metadata_eval_result.model_pretty_name",description:`<strong>model_pretty_name</strong> (<code>str</code>) &#x2014;
The name of the model in natural language.`,name:"model_pretty_name"},{anchor:"huggingface_hub.metadata_eval_result.task_pretty_name",description:`<strong>task_pretty_name</strong> (<code>str</code>) &#x2014;
The name of a task in natural language.`,name:"task_pretty_name"},{anchor:"huggingface_hub.metadata_eval_result.task_id",description:`<strong>task_id</strong> (<code>str</code>) &#x2014;
Example: automatic-speech-recognition. A task id.`,name:"task_id"},{anchor:"huggingface_hub.metadata_eval_result.metrics_pretty_name",description:`<strong>metrics_pretty_name</strong> (<code>str</code>) &#x2014;
A name for the metric in natural language. Example: Test WER.`,name:"metrics_pretty_name"},{anchor:"huggingface_hub.metadata_eval_result.metrics_id",description:`<strong>metrics_id</strong> (<code>str</code>) &#x2014;
Example: wer. A metric id from <a href="https://hf.co/metrics" rel="nofollow">https://hf.co/metrics</a>.`,name:"metrics_id"},{anchor:"huggingface_hub.metadata_eval_result.metrics_value",description:`<strong>metrics_value</strong> (<code>Any</code>) &#x2014;
The value from the metric. Example: 20.0 or &#x201C;20.0 &#xB1; 1.2&#x201D;.`,name:"metrics_value"},{anchor:"huggingface_hub.metadata_eval_result.dataset_pretty_name",description:`<strong>dataset_pretty_name</strong> (<code>str</code>) &#x2014;
The name of the dataset in natural language.`,name:"dataset_pretty_name"},{anchor:"huggingface_hub.metadata_eval_result.dataset_id",description:`<strong>dataset_id</strong> (<code>str</code>) &#x2014;
Example: common_voice. A dataset id from <a href="https://hf.co/datasets" rel="nofollow">https://hf.co/datasets</a>.`,name:"dataset_id"},{anchor:"huggingface_hub.metadata_eval_result.metrics_config",description:`<strong>metrics_config</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The name of the metric configuration used in <code>load_metric()</code>.
Example: bleurt-large-512 in <code>load_metric(&quot;bleurt&quot;, &quot;bleurt-large-512&quot;)</code>.`,name:"metrics_config"},{anchor:"huggingface_hub.metadata_eval_result.metrics_verified",description:`<strong>metrics_verified</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
If true, indicates that evaluation was generated by Hugging Face (vs. self-reported).
If a user tries to push self-reported metric results with verified=True, the push
will be rejected.`,name:"metrics_verified"},{anchor:"huggingface_hub.metadata_eval_result.dataset_config",description:`<strong>dataset_config</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Example: fr. The name of the dataset configuration used in <code>load_dataset()</code>.`,name:"dataset_config"},{anchor:"huggingface_hub.metadata_eval_result.dataset_split",description:`<strong>dataset_split</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Example: test. The name of the dataset split used in <code>load_dataset()</code>.`,name:"dataset_split"},{anchor:"huggingface_hub.metadata_eval_result.dataset_revision",description:`<strong>dataset_revision</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Example: 5503434ddd753f426f4b38109466949a1217c2bb. The name of the dataset dataset revision
used in <code>load_dataset()</code>.`,name:"dataset_revision"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard.py#L520",returnDescription:`
<p>a metadata dict with the result from a model evaluated on a dataset.</p>
`,returnType:`
<p><code>dict</code></p>
`}}),we=new ie({props:{anchor:"huggingface_hub.metadata_eval_result.example",$$slots:{default:[Tr]},$$scope:{ctx:w}}}),da=new E({props:{name:"huggingface_hub.metadata_update",anchor:"huggingface_hub.metadata_update",parameters:[{name:"repo_id",val:": str"},{name:"metadata",val:": typing.Dict"},{name:"repo_type",val:": typing.Optional[str] = None"},{name:"overwrite",val:": bool = False"},{name:"token",val:": typing.Optional[str] = None"},{name:"commit_message",val:": typing.Optional[str] = None"},{name:"commit_description",val:": typing.Optional[str] = None"},{name:"revision",val:": typing.Optional[str] = None"},{name:"create_pr",val:": bool = False"},{name:"parent_commit",val:": typing.Optional[str] = None"}],parametersDescription:[{anchor:"huggingface_hub.metadata_update.repo_id",description:`<strong>repo_id</strong> (<code>str</code>) &#x2014;
The name of the repository.`,name:"repo_id"},{anchor:"huggingface_hub.metadata_update.metadata",description:`<strong>metadata</strong> (<code>dict</code>) &#x2014;
A dictionary containing the metadata to be updated.`,name:"metadata"},{anchor:"huggingface_hub.metadata_update.repo_type",description:`<strong>repo_type</strong> (<code>str</code>, <em>optional</em>) &#x2014;
Set to <code>&quot;dataset&quot;</code> or <code>&quot;space&quot;</code> if updating to a dataset or space,
<code>None</code> or <code>&quot;model&quot;</code> if updating to a model. Default is <code>None</code>.`,name:"repo_type"},{anchor:"huggingface_hub.metadata_update.overwrite",description:`<strong>overwrite</strong> (<code>bool</code>, <em>optional</em>, defaults to <code>False</code>) &#x2014;
If set to <code>True</code> an existing field can be overwritten, otherwise
attempting to overwrite an existing field will cause an error.`,name:"overwrite"},{anchor:"huggingface_hub.metadata_update.token",description:`<strong>token</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The Hugging Face authentication token.`,name:"token"},{anchor:"huggingface_hub.metadata_update.commit_message",description:`<strong>commit_message</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The summary / title / first line of the generated commit. Defaults to
<code>f&quot;Update metdata with huggingface_hub&quot;</code>`,name:"commit_message"},{anchor:"huggingface_hub.metadata_update.commit_description",description:`<strong>commit_description</strong> (<code>str</code> <em>optional</em>) &#x2014;
The description of the generated commit`,name:"commit_description"},{anchor:"huggingface_hub.metadata_update.revision",description:`<strong>revision</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The git revision to commit from. Defaults to the head of the
<code>&quot;main&quot;</code> branch.`,name:"revision"},{anchor:"huggingface_hub.metadata_update.create_pr",description:`<strong>create_pr</strong> (<code>boolean</code>, <em>optional</em>) &#x2014;
Whether or not to create a Pull Request from <code>revision</code> with that commit.
Defaults to <code>False</code>.`,name:"create_pr"},{anchor:"huggingface_hub.metadata_update.parent_commit",description:`<strong>parent_commit</strong> (<code>str</code>, <em>optional</em>) &#x2014;
The OID / SHA of the parent commit, as a hexadecimal string. Shorthands (7 first characters) are also supported.
If specified and <code>create_pr</code> is <code>False</code>, the commit will fail if <code>revision</code> does not point to <code>parent_commit</code>.
If specified and <code>create_pr</code> is <code>True</code>, the pull request will be created from <code>parent_commit</code>.
Specifying <code>parent_commit</code> ensures the repo has not changed before committing the changes, and can be
especially useful if the repo is updated / committed to concurrently.`,name:"parent_commit"}],source:"https://github.com/huggingface/huggingface_hub/blob/vr_940/src/huggingface_hub/repocard.py#L645",returnDescription:`
<p>URL of the commit which updated the card metadata.</p>
`,returnType:`
<p><code>str</code></p>
`}}),Ee=new ie({props:{anchor:"huggingface_hub.metadata_update.example",$$slots:{default:[Ar]},$$scope:{ctx:w}}}),{c(){i=n("meta"),x=c(),f=n("h1"),l=n("a"),b=n("span"),v(t.$$.fragment),d=c(),ee=n("span"),W=g("Repository Cards"),T=c(),R=n("p"),Me=g(`The huggingface_hub library provides a Python interface to create, share, and update Model/Dataset Cards.
Visit the `),U=n("a"),A=g("dedicated documentation page"),F=g(` for a deeper view of what
Model Cards on the Hub are, and how they work under the hood. You can also check out our `),Y=n("a"),ha=g("Model Cards guide"),ua=g(` to
get a feel for how you would use these utilities in your own projects.`),de=c(),M=n("h2"),O=n("a"),La=n("span"),v(Ne.$$.fragment),Bt=c(),Ia=n("span"),zt=g("Repo Card"),ht=c(),N=n("p"),Gt=g("The "),Pa=n("code"),Jt=g("RepoCard"),Wt=g(" object is the parent class of "),fa=n("a"),Yt=g("ModelCard"),Kt=g(" and "),_a=n("a"),Qt=g("DatasetCard"),Xt=g("."),ut=c(),k=n("div"),v(Le.$$.fragment),Zt=c(),K=n("div"),v(Ie.$$.fragment),es=c(),Ha=n("p"),as=g("Initialize a RepoCard from a template. By default, it uses the default template."),ts=c(),Ua=n("p"),ss=g("Templates are Jinja2 templates that can be customized by passing keyword arguments."),ns=c(),Q=n("div"),v(Pe.$$.fragment),rs=c(),Fa=n("p"),os=g("Initialize a RepoCard from a Hugging Face Hub repo\u2019s README.md or a local filepath."),ls=c(),v(ge.$$.fragment),is=c(),me=n("div"),v(He.$$.fragment),cs=c(),Oa=n("p"),ps=g("Push a RepoCard to a Hugging Face Hub repo."),ds=c(),X=n("div"),v(Ue.$$.fragment),gs=c(),Sa=n("p"),ms=g("Save a RepoCard to a file."),hs=c(),v(he.$$.fragment),us=c(),Z=n("div"),v(Fe.$$.fragment),fs=c(),Oe=n("p"),_s=g(`Validates card against Hugging Face Hub\u2019s card validation logic.
Using this function requires access to the internet, so it is only called
internally by `),ba=n("a"),bs=g("huggingface_hub.repocard.RepoCard.push_to_hub()"),xs=g("."),vs=c(),v(ue.$$.fragment),ft=c(),L=n("p"),$s=g("The "),xa=n("a"),ys=g("CardData"),js=g(" object is the parent class of "),va=n("a"),Ds=g("ModelCardData"),Cs=g(" and "),$a=n("a"),ws=g("DatasetCardData"),Es=g("."),_t=c(),S=n("div"),v(Se.$$.fragment),ks=c(),fe=n("div"),v(Ve.$$.fragment),Rs=c(),Va=n("p"),qs=g("Converts CardData to a dict."),Ts=c(),_e=n("div"),v(Be.$$.fragment),As=c(),Ba=n("p"),Ms=g("Dumps CardData to a YAML block for inclusion in a README.md file."),bt=c(),ae=n("h2"),be=n("a"),za=n("span"),v(ze.$$.fragment),Ns=c(),Ga=n("span"),Ls=g("Model Cards"),xt=c(),te=n("div"),v(Ge.$$.fragment),Is=c(),I=n("div"),v(Je.$$.fragment),Ps=c(),ya=n("p"),Hs=g(`Initialize a ModelCard from a template. By default, it uses the default template, which can be found here:
`),We=n("a"),Us=g("https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md"),Fs=c(),Ja=n("p"),Os=g("Templates are Jinja2 templates that can be customized by passing keyword arguments."),Ss=c(),v(xe.$$.fragment),vt=c(),V=n("div"),v(Ye.$$.fragment),Vs=c(),Wa=n("p"),Bs=g("Model Card Metadata that is used by Hugging Face Hub when included at the top of your README.md"),zs=c(),v(ve.$$.fragment),$t=c(),se=n("h2"),$e=n("a"),Ya=n("span"),v(Ke.$$.fragment),Gs=c(),Ka=n("span"),Js=g("Dataset Cards"),yt=c(),ne=n("div"),v(Qe.$$.fragment),Ws=c(),P=n("div"),v(Xe.$$.fragment),Ys=c(),ja=n("p"),Ks=g(`Initialize a DatasetCard from a template. By default, it uses the default template, which can be found here:
`),Ze=n("a"),Qs=g("https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md"),Xs=c(),Qa=n("p"),Zs=g("Templates are Jinja2 templates that can be customized by passing keyword arguments."),en=c(),v(ye.$$.fragment),jt=c(),re=n("div"),v(ea.$$.fragment),an=c(),Xa=n("p"),tn=g("Dataset Card Metadata that is used by Hugging Face Hub when included at the top of your README.md"),Dt=c(),oe=n("h2"),je=n("a"),Za=n("span"),v(aa.$$.fragment),sn=c(),et=n("span"),nn=g("Utilities"),Ct=c(),B=n("div"),v(ta.$$.fragment),rn=c(),at=n("p"),on=g("Flattened representation of individual evaluation results found in model-index of Model Cards."),ln=c(),sa=n("p"),cn=g("For more information on the model-index spec, see "),na=n("a"),pn=g("https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1"),dn=g("."),wt=c(),q=n("div"),v(ra.$$.fragment),gn=c(),oa=n("p"),mn=g("Takes in a model index and returns the model name and a list of "),tt=n("code"),hn=g("huggingface_hub.EvalResult"),un=g(" objects."),fn=c(),Da=n("p"),_n=g(`A detailed spec of the model index can be found here:
`),la=n("a"),bn=g("https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1"),xn=c(),v(De.$$.fragment),Et=c(),z=n("div"),v(ia.$$.fragment),vn=c(),ca=n("p"),$n=g("Takes in given model name and list of "),st=n("code"),yn=g("huggingface_hub.EvalResult"),jn=g(` and returns a
valid model-index that will be compatible with the format expected by the
Hugging Face Hub.`),Dn=c(),v(Ce.$$.fragment),kt=c(),G=n("div"),v(pa.$$.fragment),Cn=c(),nt=n("p"),wn=g("Creates a metadata dict with the result from a model evaluated on a dataset."),En=c(),v(we.$$.fragment),Rt=c(),J=n("div"),v(da.$$.fragment),kn=c(),rt=n("p"),Rn=g("Updates the metadata in the README.md of a repository on the Hugging Face Hub."),qn=c(),v(Ee.$$.fragment),this.h()},l(e){const u=vr('[data-svelte="svelte-1phssyn"]',document.head);i=r(u,"META",{name:!0,content:!0}),u.forEach(s),x=p(e),f=r(e,"H1",{class:!0});var ga=o(f);l=r(ga,"A",{id:!0,class:!0,href:!0});var ot=o(l);b=r(ot,"SPAN",{});var lt=o(b);$(t.$$.fragment,lt),lt.forEach(s),ot.forEach(s),d=p(ga),ee=r(ga,"SPAN",{});var it=o(ee);W=m(it,"Repository Cards"),it.forEach(s),ga.forEach(s),T=p(e),R=r(e,"P",{});var le=o(R);Me=m(le,`The huggingface_hub library provides a Python interface to create, share, and update Model/Dataset Cards.
Visit the `),U=r(le,"A",{href:!0,rel:!0});var ct=o(U);A=m(ct,"dedicated documentation page"),ct.forEach(s),F=m(le,` for a deeper view of what
Model Cards on the Hub are, and how they work under the hood. You can also check out our `),Y=r(le,"A",{href:!0});var pt=o(Y);ha=m(pt,"Model Cards guide"),pt.forEach(s),ua=m(le,` to
get a feel for how you would use these utilities in your own projects.`),le.forEach(s),de=p(e),M=r(e,"H2",{class:!0});var ma=o(M);O=r(ma,"A",{id:!0,class:!0,href:!0});var dt=o(O);La=r(dt,"SPAN",{});var gt=o(La);$(Ne.$$.fragment,gt),gt.forEach(s),dt.forEach(s),Bt=p(ma),Ia=r(ma,"SPAN",{});var Nn=o(Ia);zt=m(Nn,"Repo Card"),Nn.forEach(s),ma.forEach(s),ht=p(e),N=r(e,"P",{});var ke=o(N);Gt=m(ke,"The "),Pa=r(ke,"CODE",{});var Ln=o(Pa);Jt=m(Ln,"RepoCard"),Ln.forEach(s),Wt=m(ke," object is the parent class of "),fa=r(ke,"A",{href:!0});var In=o(fa);Yt=m(In,"ModelCard"),In.forEach(s),Kt=m(ke," and "),_a=r(ke,"A",{href:!0});var Pn=o(_a);Qt=m(Pn,"DatasetCard"),Pn.forEach(s),Xt=m(ke,"."),ke.forEach(s),ut=p(e),k=r(e,"DIV",{class:!0});var H=o(k);$(Le.$$.fragment,H),Zt=p(H),K=r(H,"DIV",{class:!0});var Ca=o(K);$(Ie.$$.fragment,Ca),es=p(Ca),Ha=r(Ca,"P",{});var Hn=o(Ha);as=m(Hn,"Initialize a RepoCard from a template. By default, it uses the default template."),Hn.forEach(s),ts=p(Ca),Ua=r(Ca,"P",{});var Un=o(Ua);ss=m(Un,"Templates are Jinja2 templates that can be customized by passing keyword arguments."),Un.forEach(s),Ca.forEach(s),ns=p(H),Q=r(H,"DIV",{class:!0});var wa=o(Q);$(Pe.$$.fragment,wa),rs=p(wa),Fa=r(wa,"P",{});var Fn=o(Fa);os=m(Fn,"Initialize a RepoCard from a Hugging Face Hub repo\u2019s README.md or a local filepath."),Fn.forEach(s),ls=p(wa),$(ge.$$.fragment,wa),wa.forEach(s),is=p(H),me=r(H,"DIV",{class:!0});var Tt=o(me);$(He.$$.fragment,Tt),cs=p(Tt),Oa=r(Tt,"P",{});var On=o(Oa);ps=m(On,"Push a RepoCard to a Hugging Face Hub repo."),On.forEach(s),Tt.forEach(s),ds=p(H),X=r(H,"DIV",{class:!0});var Ea=o(X);$(Ue.$$.fragment,Ea),gs=p(Ea),Sa=r(Ea,"P",{});var Sn=o(Sa);ms=m(Sn,"Save a RepoCard to a file."),Sn.forEach(s),hs=p(Ea),$(he.$$.fragment,Ea),Ea.forEach(s),us=p(H),Z=r(H,"DIV",{class:!0});var ka=o(Z);$(Fe.$$.fragment,ka),fs=p(ka),Oe=r(ka,"P",{});var At=o(Oe);_s=m(At,`Validates card against Hugging Face Hub\u2019s card validation logic.
Using this function requires access to the internet, so it is only called
internally by `),ba=r(At,"A",{href:!0});var Vn=o(ba);bs=m(Vn,"huggingface_hub.repocard.RepoCard.push_to_hub()"),Vn.forEach(s),xs=m(At,"."),At.forEach(s),vs=p(ka),$(ue.$$.fragment,ka),ka.forEach(s),H.forEach(s),ft=p(e),L=r(e,"P",{});var Re=o(L);$s=m(Re,"The "),xa=r(Re,"A",{href:!0});var Bn=o(xa);ys=m(Bn,"CardData"),Bn.forEach(s),js=m(Re," object is the parent class of "),va=r(Re,"A",{href:!0});var zn=o(va);Ds=m(zn,"ModelCardData"),zn.forEach(s),Cs=m(Re," and "),$a=r(Re,"A",{href:!0});var Gn=o($a);ws=m(Gn,"DatasetCardData"),Gn.forEach(s),Es=m(Re,"."),Re.forEach(s),_t=p(e),S=r(e,"DIV",{class:!0});var Ra=o(S);$(Se.$$.fragment,Ra),ks=p(Ra),fe=r(Ra,"DIV",{class:!0});var Mt=o(fe);$(Ve.$$.fragment,Mt),Rs=p(Mt),Va=r(Mt,"P",{});var Jn=o(Va);qs=m(Jn,"Converts CardData to a dict."),Jn.forEach(s),Mt.forEach(s),Ts=p(Ra),_e=r(Ra,"DIV",{class:!0});var Nt=o(_e);$(Be.$$.fragment,Nt),As=p(Nt),Ba=r(Nt,"P",{});var Wn=o(Ba);Ms=m(Wn,"Dumps CardData to a YAML block for inclusion in a README.md file."),Wn.forEach(s),Nt.forEach(s),Ra.forEach(s),bt=p(e),ae=r(e,"H2",{class:!0});var Lt=o(ae);be=r(Lt,"A",{id:!0,class:!0,href:!0});var Yn=o(be);za=r(Yn,"SPAN",{});var Kn=o(za);$(ze.$$.fragment,Kn),Kn.forEach(s),Yn.forEach(s),Ns=p(Lt),Ga=r(Lt,"SPAN",{});var Qn=o(Ga);Ls=m(Qn,"Model Cards"),Qn.forEach(s),Lt.forEach(s),xt=p(e),te=r(e,"DIV",{class:!0});var It=o(te);$(Ge.$$.fragment,It),Is=p(It),I=r(It,"DIV",{class:!0});var qe=o(I);$(Je.$$.fragment,qe),Ps=p(qe),ya=r(qe,"P",{});var Tn=o(ya);Hs=m(Tn,`Initialize a ModelCard from a template. By default, it uses the default template, which can be found here:
`),We=r(Tn,"A",{href:!0,rel:!0});var Xn=o(We);Us=m(Xn,"https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md"),Xn.forEach(s),Tn.forEach(s),Fs=p(qe),Ja=r(qe,"P",{});var Zn=o(Ja);Os=m(Zn,"Templates are Jinja2 templates that can be customized by passing keyword arguments."),Zn.forEach(s),Ss=p(qe),$(xe.$$.fragment,qe),qe.forEach(s),It.forEach(s),vt=p(e),V=r(e,"DIV",{class:!0});var qa=o(V);$(Ye.$$.fragment,qa),Vs=p(qa),Wa=r(qa,"P",{});var er=o(Wa);Bs=m(er,"Model Card Metadata that is used by Hugging Face Hub when included at the top of your README.md"),er.forEach(s),zs=p(qa),$(ve.$$.fragment,qa),qa.forEach(s),$t=p(e),se=r(e,"H2",{class:!0});var Pt=o(se);$e=r(Pt,"A",{id:!0,class:!0,href:!0});var ar=o($e);Ya=r(ar,"SPAN",{});var tr=o(Ya);$(Ke.$$.fragment,tr),tr.forEach(s),ar.forEach(s),Gs=p(Pt),Ka=r(Pt,"SPAN",{});var sr=o(Ka);Js=m(sr,"Dataset Cards"),sr.forEach(s),Pt.forEach(s),yt=p(e),ne=r(e,"DIV",{class:!0});var Ht=o(ne);$(Qe.$$.fragment,Ht),Ws=p(Ht),P=r(Ht,"DIV",{class:!0});var Te=o(P);$(Xe.$$.fragment,Te),Ys=p(Te),ja=r(Te,"P",{});var An=o(ja);Ks=m(An,`Initialize a DatasetCard from a template. By default, it uses the default template, which can be found here:
`),Ze=r(An,"A",{href:!0,rel:!0});var nr=o(Ze);Qs=m(nr,"https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md"),nr.forEach(s),An.forEach(s),Xs=p(Te),Qa=r(Te,"P",{});var rr=o(Qa);Zs=m(rr,"Templates are Jinja2 templates that can be customized by passing keyword arguments."),rr.forEach(s),en=p(Te),$(ye.$$.fragment,Te),Te.forEach(s),Ht.forEach(s),jt=p(e),re=r(e,"DIV",{class:!0});var Ut=o(re);$(ea.$$.fragment,Ut),an=p(Ut),Xa=r(Ut,"P",{});var or=o(Xa);tn=m(or,"Dataset Card Metadata that is used by Hugging Face Hub when included at the top of your README.md"),or.forEach(s),Ut.forEach(s),Dt=p(e),oe=r(e,"H2",{class:!0});var Ft=o(oe);je=r(Ft,"A",{id:!0,class:!0,href:!0});var lr=o(je);Za=r(lr,"SPAN",{});var ir=o(Za);$(aa.$$.fragment,ir),ir.forEach(s),lr.forEach(s),sn=p(Ft),et=r(Ft,"SPAN",{});var cr=o(et);nn=m(cr,"Utilities"),cr.forEach(s),Ft.forEach(s),Ct=p(e),B=r(e,"DIV",{class:!0});var Ta=o(B);$(ta.$$.fragment,Ta),rn=p(Ta),at=r(Ta,"P",{});var pr=o(at);on=m(pr,"Flattened representation of individual evaluation results found in model-index of Model Cards."),pr.forEach(s),ln=p(Ta),sa=r(Ta,"P",{});var Ot=o(sa);cn=m(Ot,"For more information on the model-index spec, see "),na=r(Ot,"A",{href:!0,rel:!0});var dr=o(na);pn=m(dr,"https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1"),dr.forEach(s),dn=m(Ot,"."),Ot.forEach(s),Ta.forEach(s),wt=p(e),q=r(e,"DIV",{class:!0});var Ae=o(q);$(ra.$$.fragment,Ae),gn=p(Ae),oa=r(Ae,"P",{});var St=o(oa);mn=m(St,"Takes in a model index and returns the model name and a list of "),tt=r(St,"CODE",{});var gr=o(tt);hn=m(gr,"huggingface_hub.EvalResult"),gr.forEach(s),un=m(St," objects."),St.forEach(s),fn=p(Ae),Da=r(Ae,"P",{});var Mn=o(Da);_n=m(Mn,`A detailed spec of the model index can be found here:
`),la=r(Mn,"A",{href:!0,rel:!0});var mr=o(la);bn=m(mr,"https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1"),mr.forEach(s),Mn.forEach(s),xn=p(Ae),$(De.$$.fragment,Ae),Ae.forEach(s),Et=p(e),z=r(e,"DIV",{class:!0});var Aa=o(z);$(ia.$$.fragment,Aa),vn=p(Aa),ca=r(Aa,"P",{});var Vt=o(ca);$n=m(Vt,"Takes in given model name and list of "),st=r(Vt,"CODE",{});var hr=o(st);yn=m(hr,"huggingface_hub.EvalResult"),hr.forEach(s),jn=m(Vt,` and returns a
valid model-index that will be compatible with the format expected by the
Hugging Face Hub.`),Vt.forEach(s),Dn=p(Aa),$(Ce.$$.fragment,Aa),Aa.forEach(s),kt=p(e),G=r(e,"DIV",{class:!0});var Ma=o(G);$(pa.$$.fragment,Ma),Cn=p(Ma),nt=r(Ma,"P",{});var ur=o(nt);wn=m(ur,"Creates a metadata dict with the result from a model evaluated on a dataset."),ur.forEach(s),En=p(Ma),$(we.$$.fragment,Ma),Ma.forEach(s),Rt=p(e),J=r(e,"DIV",{class:!0});var Na=o(J);$(da.$$.fragment,Na),kn=p(Na),rt=r(Na,"P",{});var fr=o(rt);Rn=m(fr,"Updates the metadata in the README.md of a repository on the Hugging Face Hub."),fr.forEach(s),qn=p(Na),$(Ee.$$.fragment,Na),Na.forEach(s),this.h()},h(){h(i,"name","hf:doc:metadata"),h(i,"content",JSON.stringify(Nr)),h(l,"id","repository-cards"),h(l,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(l,"href","#repository-cards"),h(f,"class","relative group"),h(U,"href","https://huggingface.co/docs/hub/models-cards"),h(U,"rel","nofollow"),h(Y,"href","../how-to-model-cards"),h(O,"id","huggingface_hub.repocard.RepoCard"),h(O,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(O,"href","#huggingface_hub.repocard.RepoCard"),h(M,"class","relative group"),h(fa,"href","/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.ModelCard"),h(_a,"href","/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.DatasetCard"),h(K,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(Q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(me,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(X,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(ba,"href","/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.repocard.RepoCard.push_to_hub"),h(Z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(k,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(xa,"href","/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.CardData"),h(va,"href","/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.ModelCardData"),h($a,"href","/docs/huggingface_hub/pr_940/en/package_reference/cards#huggingface_hub.DatasetCardData"),h(fe,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(_e,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(S,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(be,"id","huggingface_hub.ModelCard"),h(be,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(be,"href","#huggingface_hub.ModelCard"),h(ae,"class","relative group"),h(We,"href","https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md"),h(We,"rel","nofollow"),h(I,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(te,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(V,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h($e,"id","huggingface_hub.DatasetCard"),h($e,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h($e,"href","#huggingface_hub.DatasetCard"),h(se,"class","relative group"),h(Ze,"href","https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md"),h(Ze,"rel","nofollow"),h(P,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(ne,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(re,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(je,"id","huggingface_hub.EvalResult"),h(je,"class","header-link block pr-1.5 text-lg no-hover:hidden with-hover:absolute with-hover:p-1.5 with-hover:opacity-0 with-hover:group-hover:opacity-100 with-hover:right-full"),h(je,"href","#huggingface_hub.EvalResult"),h(oe,"class","relative group"),h(na,"href","https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1"),h(na,"rel","nofollow"),h(B,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(la,"href","https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1"),h(la,"rel","nofollow"),h(q,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(z,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(G,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8"),h(J,"class","docstring border-l-2 border-t-2 pl-4 pt-3.5 border-gray-100 rounded-tl-xl mb-6 mt-8")},m(e,u){a(document.head,i),_(e,x,u),_(e,f,u),a(f,l),a(l,b),y(t,b,null),a(f,d),a(f,ee),a(ee,W),_(e,T,u),_(e,R,u),a(R,Me),a(R,U),a(U,A),a(R,F),a(R,Y),a(Y,ha),a(R,ua),_(e,de,u),_(e,M,u),a(M,O),a(O,La),y(Ne,La,null),a(M,Bt),a(M,Ia),a(Ia,zt),_(e,ht,u),_(e,N,u),a(N,Gt),a(N,Pa),a(Pa,Jt),a(N,Wt),a(N,fa),a(fa,Yt),a(N,Kt),a(N,_a),a(_a,Qt),a(N,Xt),_(e,ut,u),_(e,k,u),y(Le,k,null),a(k,Zt),a(k,K),y(Ie,K,null),a(K,es),a(K,Ha),a(Ha,as),a(K,ts),a(K,Ua),a(Ua,ss),a(k,ns),a(k,Q),y(Pe,Q,null),a(Q,rs),a(Q,Fa),a(Fa,os),a(Q,ls),y(ge,Q,null),a(k,is),a(k,me),y(He,me,null),a(me,cs),a(me,Oa),a(Oa,ps),a(k,ds),a(k,X),y(Ue,X,null),a(X,gs),a(X,Sa),a(Sa,ms),a(X,hs),y(he,X,null),a(k,us),a(k,Z),y(Fe,Z,null),a(Z,fs),a(Z,Oe),a(Oe,_s),a(Oe,ba),a(ba,bs),a(Oe,xs),a(Z,vs),y(ue,Z,null),_(e,ft,u),_(e,L,u),a(L,$s),a(L,xa),a(xa,ys),a(L,js),a(L,va),a(va,Ds),a(L,Cs),a(L,$a),a($a,ws),a(L,Es),_(e,_t,u),_(e,S,u),y(Se,S,null),a(S,ks),a(S,fe),y(Ve,fe,null),a(fe,Rs),a(fe,Va),a(Va,qs),a(S,Ts),a(S,_e),y(Be,_e,null),a(_e,As),a(_e,Ba),a(Ba,Ms),_(e,bt,u),_(e,ae,u),a(ae,be),a(be,za),y(ze,za,null),a(ae,Ns),a(ae,Ga),a(Ga,Ls),_(e,xt,u),_(e,te,u),y(Ge,te,null),a(te,Is),a(te,I),y(Je,I,null),a(I,Ps),a(I,ya),a(ya,Hs),a(ya,We),a(We,Us),a(I,Fs),a(I,Ja),a(Ja,Os),a(I,Ss),y(xe,I,null),_(e,vt,u),_(e,V,u),y(Ye,V,null),a(V,Vs),a(V,Wa),a(Wa,Bs),a(V,zs),y(ve,V,null),_(e,$t,u),_(e,se,u),a(se,$e),a($e,Ya),y(Ke,Ya,null),a(se,Gs),a(se,Ka),a(Ka,Js),_(e,yt,u),_(e,ne,u),y(Qe,ne,null),a(ne,Ws),a(ne,P),y(Xe,P,null),a(P,Ys),a(P,ja),a(ja,Ks),a(ja,Ze),a(Ze,Qs),a(P,Xs),a(P,Qa),a(Qa,Zs),a(P,en),y(ye,P,null),_(e,jt,u),_(e,re,u),y(ea,re,null),a(re,an),a(re,Xa),a(Xa,tn),_(e,Dt,u),_(e,oe,u),a(oe,je),a(je,Za),y(aa,Za,null),a(oe,sn),a(oe,et),a(et,nn),_(e,Ct,u),_(e,B,u),y(ta,B,null),a(B,rn),a(B,at),a(at,on),a(B,ln),a(B,sa),a(sa,cn),a(sa,na),a(na,pn),a(sa,dn),_(e,wt,u),_(e,q,u),y(ra,q,null),a(q,gn),a(q,oa),a(oa,mn),a(oa,tt),a(tt,hn),a(oa,un),a(q,fn),a(q,Da),a(Da,_n),a(Da,la),a(la,bn),a(q,xn),y(De,q,null),_(e,Et,u),_(e,z,u),y(ia,z,null),a(z,vn),a(z,ca),a(ca,$n),a(ca,st),a(st,yn),a(ca,jn),a(z,Dn),y(Ce,z,null),_(e,kt,u),_(e,G,u),y(pa,G,null),a(G,Cn),a(G,nt),a(nt,wn),a(G,En),y(we,G,null),_(e,Rt,u),_(e,J,u),y(da,J,null),a(J,kn),a(J,rt),a(rt,Rn),a(J,qn),y(Ee,J,null),qt=!0},p(e,[u]){const ga={};u&2&&(ga.$$scope={dirty:u,ctx:e}),ge.$set(ga);const ot={};u&2&&(ot.$$scope={dirty:u,ctx:e}),he.$set(ot);const lt={};u&2&&(lt.$$scope={dirty:u,ctx:e}),ue.$set(lt);const it={};u&2&&(it.$$scope={dirty:u,ctx:e}),xe.$set(it);const le={};u&2&&(le.$$scope={dirty:u,ctx:e}),ve.$set(le);const ct={};u&2&&(ct.$$scope={dirty:u,ctx:e}),ye.$set(ct);const pt={};u&2&&(pt.$$scope={dirty:u,ctx:e}),De.$set(pt);const ma={};u&2&&(ma.$$scope={dirty:u,ctx:e}),Ce.$set(ma);const dt={};u&2&&(dt.$$scope={dirty:u,ctx:e}),we.$set(dt);const gt={};u&2&&(gt.$$scope={dirty:u,ctx:e}),Ee.$set(gt)},i(e){qt||(j(t.$$.fragment,e),j(Ne.$$.fragment,e),j(Le.$$.fragment,e),j(Ie.$$.fragment,e),j(Pe.$$.fragment,e),j(ge.$$.fragment,e),j(He.$$.fragment,e),j(Ue.$$.fragment,e),j(he.$$.fragment,e),j(Fe.$$.fragment,e),j(ue.$$.fragment,e),j(Se.$$.fragment,e),j(Ve.$$.fragment,e),j(Be.$$.fragment,e),j(ze.$$.fragment,e),j(Ge.$$.fragment,e),j(Je.$$.fragment,e),j(xe.$$.fragment,e),j(Ye.$$.fragment,e),j(ve.$$.fragment,e),j(Ke.$$.fragment,e),j(Qe.$$.fragment,e),j(Xe.$$.fragment,e),j(ye.$$.fragment,e),j(ea.$$.fragment,e),j(aa.$$.fragment,e),j(ta.$$.fragment,e),j(ra.$$.fragment,e),j(De.$$.fragment,e),j(ia.$$.fragment,e),j(Ce.$$.fragment,e),j(pa.$$.fragment,e),j(we.$$.fragment,e),j(da.$$.fragment,e),j(Ee.$$.fragment,e),qt=!0)},o(e){D(t.$$.fragment,e),D(Ne.$$.fragment,e),D(Le.$$.fragment,e),D(Ie.$$.fragment,e),D(Pe.$$.fragment,e),D(ge.$$.fragment,e),D(He.$$.fragment,e),D(Ue.$$.fragment,e),D(he.$$.fragment,e),D(Fe.$$.fragment,e),D(ue.$$.fragment,e),D(Se.$$.fragment,e),D(Ve.$$.fragment,e),D(Be.$$.fragment,e),D(ze.$$.fragment,e),D(Ge.$$.fragment,e),D(Je.$$.fragment,e),D(xe.$$.fragment,e),D(Ye.$$.fragment,e),D(ve.$$.fragment,e),D(Ke.$$.fragment,e),D(Qe.$$.fragment,e),D(Xe.$$.fragment,e),D(ye.$$.fragment,e),D(ea.$$.fragment,e),D(aa.$$.fragment,e),D(ta.$$.fragment,e),D(ra.$$.fragment,e),D(De.$$.fragment,e),D(ia.$$.fragment,e),D(Ce.$$.fragment,e),D(pa.$$.fragment,e),D(we.$$.fragment,e),D(da.$$.fragment,e),D(Ee.$$.fragment,e),qt=!1},d(e){s(i),e&&s(x),e&&s(f),C(t),e&&s(T),e&&s(R),e&&s(de),e&&s(M),C(Ne),e&&s(ht),e&&s(N),e&&s(ut),e&&s(k),C(Le),C(Ie),C(Pe),C(ge),C(He),C(Ue),C(he),C(Fe),C(ue),e&&s(ft),e&&s(L),e&&s(_t),e&&s(S),C(Se),C(Ve),C(Be),e&&s(bt),e&&s(ae),C(ze),e&&s(xt),e&&s(te),C(Ge),C(Je),C(xe),e&&s(vt),e&&s(V),C(Ye),C(ve),e&&s($t),e&&s(se),C(Ke),e&&s(yt),e&&s(ne),C(Qe),C(Xe),C(ye),e&&s(jt),e&&s(re),C(ea),e&&s(Dt),e&&s(oe),C(aa),e&&s(Ct),e&&s(B),C(ta),e&&s(wt),e&&s(q),C(ra),C(De),e&&s(Et),e&&s(z),C(ia),C(Ce),e&&s(kt),e&&s(G),C(pa),C(we),e&&s(Rt),e&&s(J),C(da),C(Ee)}}}const Nr={local:"repository-cards",sections:[{local:"huggingface_hub.repocard.RepoCard",title:"Repo Card"},{local:"huggingface_hub.ModelCard",title:"Model Cards"},{local:"huggingface_hub.DatasetCard",title:"Dataset Cards"},{local:"huggingface_hub.EvalResult",title:"Utilities"}],title:"Repository Cards"};function Lr(w){return $r(()=>{new URLSearchParams(window.location.search).get("fw")}),[]}class Sr extends _r{constructor(i){super();br(this,i,Lr,Mr,xr,{})}}export{Sr as default,Nr as metadata};
