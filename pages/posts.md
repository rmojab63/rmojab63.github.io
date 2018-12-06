---
layout: page
---

{% assign postsByYear = site.posts | where: "htmllang", "en" | group_by_exp:"post", "post.date | date: '%Y'" %}
{% for year in postsByYear %}
## {{ year.name }}
    {% assign postsByMonth = year.items | group_by_exp:"post", "post.date | date: '%B'" %}
    {% for month in postsByMonth %}
### {{ month.name }}
        {% for post in month.items %}
* [{{ post.title }}]({{ post.url }})
        {% endfor %}
    {% endfor %}
{% endfor %}