# Homebrew Governance Archives

{% assign governance_pages = site.pages | where: "category", "governance-archives" %}

{% for item in governance_pages -%}
- [{{ item.title }}]({{ item.url }})
{% endfor %}
