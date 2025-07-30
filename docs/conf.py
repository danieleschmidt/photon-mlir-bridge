# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path

# Add project root to path for autodoc
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

# -- Project information -----------------------------------------------------
project = 'Photon MLIR Bridge'
copyright = '2025, Daniel Schmidt'
author = 'Daniel Schmidt'
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.duration',
    'sphinx.ext.autosectionlabel',
    'myst_parser',
    'breathe',
    'exhale',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'sphinxcontrib.plantuml',
    'sphinxcontrib.bibtex',
    'sphinx_design',
    'sphinx_togglebutton',
    'sphinx_external_toc',
]

# Templates path
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980b9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Custom CSS and JS
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
html_js_files = [
    'custom.js',
]

# HTML context for custom variables
html_context = {
    'display_github': True,
    'github_user': 'yourusername',
    'github_repo': 'photon-mlir-bridge',
    'github_version': 'main',
    'conf_py_path': '/docs/',
    'source_suffix': '.md',
}

# HTML options
html_title = f"{project} v{version}"
html_short_title = "Photon MLIR"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True
html_last_updated_fmt = '%b %d, %Y'
html_use_smartypants = True
html_sidebars = {
    '**': [
        'relations.html',
        'searchbox.html',
        'localtoc.html',
    ]
}

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    'preamble': r'''
    \usepackage{charter}
    \usepackage[defaultsans]{lato}
    \usepackage{inconsolata}
    ''',
    'fncychap': '\\usepackage[Bjornstrup]{fncychap}',
    'printindex': '\\footnotesize\\raggedright\\printindex',
}

latex_documents = [
    ('index', 'photon-mlir-bridge.tex', 'Photon MLIR Bridge Documentation',
     'Daniel Schmidt', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    ('index', 'photon-mlir-bridge', 'Photon MLIR Bridge Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    ('index', 'photon-mlir-bridge', 'Photon MLIR Bridge Documentation',
     author, 'photon-mlir-bridge', 'MLIR compiler for photonic accelerators.',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------
epub_title = project
epub_exclude_files = ['search.html']

# -- Extension configuration --------------------------------------------------

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'mlir': ('https://mlir.llvm.org/docs/', None),
    'llvm': ('https://llvm.org/docs/', None),
}

# MyST parser settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3
myst_footnote_transition = True
myst_dmath_double_inline = True

# Breathe configuration (for C++ API docs)
breathe_projects = {
    "photon-mlir-bridge": "_build/doxygen/xml"
}
breathe_default_project = "photon-mlir-bridge"
breathe_default_members = ('members', 'undoc-members')

# Exhale configuration (for C++ API docs)
exhale_args = {
    "containmentFolder": "./api/cpp",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "C++ API Reference",
    "doxygenStripFromPath": "..",
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "exhaleUseDoxyfile": True,
    "exhaleDoxygenStdin": """
    INPUT            = ../include ../src
    GENERATE_XML     = YES
    XML_OUTPUT       = xml
    RECURSIVE        = YES
    EXTRACT_ALL      = YES
    EXTRACT_PRIVATE  = YES
    EXTRACT_STATIC   = YES
    CASE_SENSE_NAMES = YES
    SORT_MEMBER_DOCS = NO
    GENERATE_TREEVIEW = YES
    DISABLE_INDEX    = NO
    FULL_PATH_NAMES  = YES
    STRIP_FROM_PATH  = ../
    JAVADOC_AUTOBRIEF = YES
    QT_AUTOBRIEF     = YES
    MULTILINE_CPP_IS_BRIEF = YES
    INHERIT_DOCS     = YES
    SEPARATE_MEMBER_PAGES = NO
    TAB_SIZE         = 2
    ALIASES         += "rst=\\verbatim embed:rst:leading-asterisk"
    ALIASES         += "endrst=\\endverbatim"
    MACRO_EXPANSION  = YES
    EXPAND_ONLY_PREDEF = YES
    PREDEFINED      += "DOXYGEN_SHOULD_SKIP_THIS"
    """,
}

# PlantUML configuration
plantuml = 'java -jar /usr/share/plantuml/plantuml.jar'
plantuml_output_format = 'svg'

# Bibliography configuration
bibtex_bibfiles = ['references.bib']
bibtex_reference_style = 'author_year'

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True

# Todo extension settings
todo_include_todos = True
todo_emit_warnings = True

# Coverage settings
coverage_show_missing_items = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom roles and directives
def setup(app):
    """Custom Sphinx setup."""
    # Add custom CSS class for photonic computing terms
    app.add_css_file('photonic.css')
    
    # Custom role for photonic terms
    from docutils import nodes
    from docutils.parsers.rst import roles
    
    def photonic_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        """Custom role for photonic computing terms."""
        options['classes'] = ['photonic-term']
        node = nodes.inline(rawtext, text, **options)
        return [node], []
    
    roles.register_local_role('photonic', photonic_role)
    
    # Add build info
    app.add_config_value('build_date', None, 'env')
    app.add_config_value('git_commit', None, 'env')

# Build information
import datetime
html_context.update({
    'build_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'commit_hash': os.environ.get('GITHUB_SHA', 'unknown')[:8],
})

# Version info
html_context.update({
    'version': version,
    'release': release,
})

# Search configuration
html_search_language = 'en'
html_search_options = {'type': 'default'}

# Social media and analytics
html_theme_options.update({
    'analytics_anonymize_ip': False,
    'logo_only': False,
    'display_version': True,
})

# External links
extlinks = {
    'issue': ('https://github.com/yourusername/photon-mlir-bridge/issues/%s', 'issue #%s'),
    'pr': ('https://github.com/yourusername/photon-mlir-bridge/pull/%s', 'PR #%s'),
    'commit': ('https://github.com/yourusername/photon-mlir-bridge/commit/%s', 'commit %s'),
}

# Suppress warnings
suppress_warnings = [
    'myst.header',  # Suppress header warnings in MyST
]

# Language settings
language = 'en'
locale_dirs = ['locale/']
gettext_compact = False

# PDF output settings
latex_show_pagerefs = True
latex_show_urls = 'footnote'

# ePub settings
epub_scheme = 'URL'
epub_identifier = 'https://photon-mlir.readthedocs.io/'
epub_publisher = 'Photon MLIR Bridge Team'
epub_copyright = copyright

# Linkcheck settings
linkcheck_ignore = [
    r'http://localhost:\d+/',
    r'https://127\.0\.0\.1:\d+/',
    r'.*github\.com/yourusername/.*',  # Will be updated with real repo
]

linkcheck_timeout = 5
linkcheck_retries = 2