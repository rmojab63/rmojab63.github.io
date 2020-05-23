---
layout: default
---
 
{% assign postsByYear = site.posts | where: "posttype", "1" | where: "htmllang", "fa" | group_by_exp:"post", "post.year" %}
{% for year in postsByYear %}
<h2 dir="rtl">
{{ year.name }}
</h2>
    {% assign postsByMonth = year.items | group_by_exp:"post", "post.month" %}
    {% for month in postsByMonth %}
<h3 dir="rtl">
{{ month.name }}
</h3>
<div dir="rtl">
<ul style="margin-right: 20px;">
{% for post in month.items %}
<li><div> <a style="font-weight:bold" href="{{ post.url }}" >{{ post.title }}</a> 
<p>
{{ post.content | newline_to_br | strip_newlines | split: '<br />' | first | strip_html | append: " [. . .]" }}</p></div></li>
	{% endfor %}
</ul>
</div>
    {% endfor %}
{% endfor %}