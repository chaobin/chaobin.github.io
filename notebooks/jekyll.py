# modification of config created here: https://gist.github.com/cscorley/9144544
try:
    from urllib.parse import quote  # Py 3
except ImportError:
    from urllib2 import quote  # Py 2
import os
import sys

BLOG_DIR = os.environ['BLOG_DIR']
NOTEBOOK_DIR = os.path.join(BLOG_DIR, 'notebooks')

f = None
for arg in sys.argv:
    if arg.endswith('.ipynb'):
        filename = os.path.basename(arg)
        basename = filename.split(os.path.extsep)[0]
        break

POST_DIR = os.sep.join((NOTEBOOK_DIR, 'temp', basename))
c = get_config()
c.NbConvertApp.export_format = 'markdown'
c.MarkdownExporter.template_path = [NOTEBOOK_DIR]
c.MarkdownExporter.template_file = 'jekyll'
c.NbConvertApp.output_base = os.path.join(POST_DIR, basename)
#c.Application.verbose_crash=True

# modify this function to point your images to a custom path
# by default this saves all images to a directory 'images' in the root of the blog directory
def path2support(path):
    """Turn a file path into a URL"""
    path_img = os.path.join('/images', basename)
    return os.path.join(path_img, os.path.basename(path))

c.MarkdownExporter.filters = {'path2support': path2support}

c.FilesWriter.build_directory = os.path.join(NOTEBOOK_DIR, 'build')
