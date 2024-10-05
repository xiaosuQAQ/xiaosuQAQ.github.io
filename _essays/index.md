---
layout: article
titles: 
  # @start locale config
  en      : &EN      Essays
  en-GB   : *EN
  en-US   : *EN
  en-CA   : *EN
  en-AU   : *EN
  zh-Hans : &ZH_HANS  
  zh      : *ZH_HANS
  zh-CN   : *ZH_HANS  随笔
  zh-SG   : *ZH_HANS
  zh-Hant : &ZH_HANT  關於
---
<ul>
  {% for essay in site.essays %}
    <li><a href="{{ essay.url }}">{{ essay.title }}</a></li>
  {% endfor %}
</ul>
