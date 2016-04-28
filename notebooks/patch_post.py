#!/usr/bin/env python
import os
import sys
import glob
import re
from string import Template
import shutil


BLOG_DIR = os.environ['BLOG_DIR']
NOTEBOOK_DIR = os.sep.join((BLOG_DIR, 'notebooks', 'temp'))
KATEX_TEMPLATE = Template('''
{% raw %}
<div class="equation" data="$latex"></div>
{% endraw %}
''')

def confirm(msg):
    if input(msg) != "Y":
        raise SystemExit()

def INFO(msg):
    print("[INFO] %s" % msg)

def basename_no_ext(filename):
    '''Minus the ext of the basename of a filename.'''
    name = os.path.basename(filename).split(os.extsep)[0]
    return name

def get_nbconvert_output_path(fullname_ipynb):
    name = basename_no_ext(fullname_ipynb)
    path_markdown = os.path.join(NOTEBOOK_DIR, name)
    return path_markdown

def patch_post(fullname, pattern=re.compile(r"\$\$.*\$\$")):
    # back up
    name_bk = fullname + '_bk'
    INFO("%s backed up as %s" % (fullname, name_bk))
    shutil.copy(fullname, name_bk)
    def replace(line):
        latex = pattern.search(line)
        if latex is None: return line
        latex = latex.group()
        latex_expression = latex.split("$$")[1]
        katex = KATEX_TEMPLATE.substitute(latex=latex_expression)
        _line = line.replace(latex, katex)
        return _line
    with open(name_bk) as post:
        with open(fullname, 'w') as output:
            for line in post:
                newline = replace(line)
                output.write(newline)
    INFO("%s patched" % fullname)

def rename_and_copy_over_post(fullname):
    basename = os.path.basename(fullname)
    path_post = os.path.join(BLOG_DIR, '_posts')
    dest_name = os.path.join(path_post, basename)
    if os.path.exists(dest_name):
        confirm("%s exists, override? (Y/n) " % dest_name)
        os.remove(dest_name)
    shutil.copy(fullname, path_post)
    INFO("%s copied to _post" % fullname)

def copy_images_over(fullname):
    base_dir = os.path.dirname(fullname)
    filename_base = basename_no_ext(fullname)
    path_images = os.sep.join((BLOG_DIR, 'images', filename_base))
    if not os.path.exists(path_images): os.mkdir(path_images)
    for img in glob.glob('%s*.png' % os.path.join(base_dir, filename_base)):
        INFO("copying %s to %s" % (img, path_images))
        shutil.copy(img, path_images)

def main():
    fullname_ipynb = sys.argv[1]
    path_output = get_nbconvert_output_path(fullname_ipynb)
    basename = basename_no_ext(fullname_ipynb)
    fullname_markdown = os.path.join(path_output, basename) + '.md'
    patch_post(fullname_markdown)
    rename_and_copy_over_post(fullname_markdown)
    copy_images_over(fullname_markdown)

if __name__ == '__main__':
    main()
