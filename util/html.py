"""
HTML result visualization utility
Copyright (C) 2020 Roy Or-El. All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Modified by Xingyuangfy 2025
Modifications include:
- Code formatting and documentation
- Integration with AT-GAN project
"""

import dominate
import math
from dominate.tags import *
import os


class HTML:
    """HTML Class for creating an HTML visualization page.
    
    This class provides functionality to create a webpage that displays images
    and their corresponding text descriptions in a grid layout.
    """
    def __init__(self, web_dir, title, refresh=0):
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        return self.img_dir

    def add_header(self, str):
        with self.doc:
            h3(str)

    def add_table(self, border=1):
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def add_images(self, ims, txts, links, width=512, cols=0):
        imNum = len(ims)
        self.add_table()
        with self.t:
            if cols == 0:
                with tr():
                    for im, txt, link in zip(ims, txts, links):
                        with td(style="word-wrap: break-word;", halign="center", valign="top"):
                            with p():
                                with a(href=os.path.join('images', link)):
                                    img(style="width:%dpx" % width, src=os.path.join('images', im))
                                br()
                                p(txt)
            else:
                rows = int(math.ceil(float(imNum) / float(cols)))
                for i in range(rows):
                    with tr():
                        for j in range(cols):
                            im = ims[i*cols + j]
                            txt = txts[i*cols + j]
                            link = links[i*cols + j]
                            with td(style="word-wrap: break-word;", halign="center", valign="top"):
                                with p():
                                    with a(href=os.path.join('images', link)):
                                        img(style="width:%dpx" % width, src=os.path.join('images', im))
                                    br()
                                    p(txt)

    def save(self):
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims = []
    txts = []
    links = []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
