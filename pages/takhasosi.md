---
layout: default
---

{% assign postsByYear = site.posts | where: "posttype", "2" | where: "htmllang", "fa" | group_by_exp:"post", "post.year" %}
{% for year in postsByYear %}

<h2 dir="rtl">
{{ year.name }}
</h2>
    {% assign postsBySeason = year.items | group_by_exp:"post", "post.season" %}
    {% for season in postsBySeason %}
	{% if season.size == 0 %}
  {% continue %}
{% endif %}
<h3 dir="rtl">
{{ season.name }}
</h3>
<div dir="rtl">
<ul style="margin-right: 20px;">
{% for post in season.items %}
<li><div> <a style="font-weight:bold" href="{{ post.url }}" >{{ post.heading }}</a> 
<p>
{{ post.content | newline_to_br | strip_newlines | split: '<br />' | first | strip_html | append: " [. . .]" }}</p></div></li>
	{% endfor %}
</ul>
</div>
    {% endfor %}
{% endfor %}