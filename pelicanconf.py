#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = 'Kimi Yuan'
SITENAME = "Kimi's Blog"
SITEURL = ''
SITETITLE = "Kimi's Blog"
SITESUBTITLE = 'Developer, Interested in Python & Data Science'
SITEDESCRIPTION = 'Kimi\'s Thoughts and Writings'
SITELOGO = '/images/profile.jpg'
#SITELOGO = SITEURL + '/images/profile.jpg'
#FAVICON = SITEURL + '/images/favicon.ico'

PATH = 'content'
STATIC_PATHS = ['pages', 'images','extra', 'extra/robots.txt']

EXTRA_PATH_METADATA = {
    'extra/robots.txt': {'path': 'robots.txt'},
    }


THEME = 'pelican-themes/Flex_blue'
DATE_FORMATS = {'en': '%a %d %b %Y'}
ARTICLE_PATHS = ['pages']
ARTICLE_URL = ('{slug}.html')
ARTICLE_SAVE_AS = ('{slug}.html')
PAGE_URL = ('{slug}.html')
#PAGE_SAVE_AS = ('{slug}.html')
#AUTHOR_URL = ('author/{name}/')
#TAG_URL = ('tag/{name}/')
# Display pages list on the top menu
#DISPLAY_PAGES_ON_MENU = True

# Display categories list on the top menu
DISPLAY_CATEGORIES_ON_MENU = True

# Display categories list as a submenu of the top menu
#DISPLAY_CATEGORIES_ON_SUBMENU = False

# Display the category in the article's info
#DISPLAY_CATEGORIES_ON_POSTINFO = False

# Display the author in the article's info
#DISPLAY_AUTHOR_ON_POSTINFO = False

# Display the search form
#DISPLAY_SEARCH_FORM = False

# Sort pages list by a given attribute
#PAGES_SORT_ATTRIBUTE (Title)

# Display the "Fork me on Github" banner
GITHUB_URL = None

LOAD_CONTENT_CACHE = False

TIMEZONE = 'Asia/Shanghai'

DEFAULT_LANG = 'en'



# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
#LINKS = (('Pelican', 'http://getpelican.com/'),
#         ('Python.org', 'http://python.org/'),
#         ('Jinja2', 'http://jinja.pocoo.org/'),
#         ('You can modify those links in your config file', '#'),)

# Social widget
SOCIAL = (('envelope-o','mailto:midwestdc914@gmail.com'),
          ('linkedin', 'https://linkedin.com/in/kimi-yuan'),
          ('github', 'https://github.com/yx0906'),
          ('twitter', 'https://twitter.com/Apocalypse_Manu')
          )

DEFAULT_PAGINATION = 10

DEFAULT_DATE = 'fs'

# Theme
COPYRIGHT_YEAR = 2016

USE_FOLDER_AS_CATEGORY = False
MAIN_MENU = True
MENUITEMS = (('Categories', '/categories.html'),
             ('Archives', '/archives.html'),
             ('Tags', '/tags.html'),)

EXTRA_PATH_METADATA = {
    'extra/custom.css': {'path': 'static/custom.css'},
}

CUSTOM_CSS = 'static/custom.css'

PYGMENTS_STYLE = 'github'

# Plugin
PLUGIN_PATHS = ['pelican-plugins']
#PLUGINS = ['assets', 'sitemap', 'gravatar','extract_toc']
#PLUGINS = ['better_tables']
PLUGINS = ['neighbors', 'sitemap', 'ipynb.markup','extract_toc', 'related_posts', 'render_math', 'filetime_from_git']
MARKUP = ('md', 'ipynb')
#MD_EXTENSIONS = ['codehilite(css_class=codehilite)', 'extra', 'toc', 'meta']
MARKDOWN = {
        'extension_configs': {
            'markdown.extensions.codehilite' : {'css_class': 'codehilite',
                #'noclasses': True, 'pygments_style': 'vim'
            },
            'markdown.extensions.extra': {},
            'markdown.extensions.meta': {},
            'markdown.extensions.toc': {}
        }
}
RELATED_POSTS_MAX = 10
#MATH_JAX = {'align':'left'}
#MATH_JAX = {'tex_extensions': ['color.js','mhchem.js']}
# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True

SITEMAP = {
    'format': 'xml',
    'priorities': {
        'articles': 0.6,
        'indexes': 0.6,
        'pages': 0.5,
    },
    'changefreqs': {
        'articles': 'monthly',
        'indexes': 'daily',
        'pages': 'monthly',
    }
}
