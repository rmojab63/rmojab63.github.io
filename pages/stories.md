---
layout: default
---
{% assign posts = site.posts | where: "posttype", "5" %}
{% for post in posts %}
<img src="{{ post.image }}" alt="image" style="display:block;margin-left:auto;margin-right:auto;width:80%;">
<div style="text-align:center;margin-bottom:2cm"> <a style="font-weight:bold" href="{{ post.url }}" >{{ post.title }}</a> </div>
{% endfor %}