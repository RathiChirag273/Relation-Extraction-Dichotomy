from PIL import Image, ImageDraw, ImageFont

"""
Visualize text in a single sample document

Parameters
----------
doc : dict
    Python dictionary (parsed json) of a single sample from the DocRED dataset.

Returns
-------
PIL.Image
    A PIL image object
"""
def vis_text(doc : dict, img_width=0, paddings=(50, 50), fsize_t=30, fsize_c=15) -> Image:
    img_height = paddings[1] + len(doc['sents']) * (fsize_c + 10) + 100
    
    # create fonts for title and content
    title_font = ImageFont.truetype("arial.ttf", fsize_t)
    content_font = ImageFont.truetype("arial.ttf", fsize_c)

    # calculate font width and line width
    x, y = paddings
    if img_width == 0:
        max_line_width = max([content_font.font.getsize(" ".join(sent))[0][0] for sent in doc['sents']])
        img_width = x*2 + max_line_width

    # create image and overlay image
    img = Image.new("RGBA", (img_width, img_height), "white")
    overlay = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    overlay_draw = ImageDraw.Draw(overlay)
    
    # draw title    
    draw.text((x, y), doc['title'], fill="black", font=title_font)

    # start drawing text body (sentences)
    y += fsize_t + 10
    sy = y

    # draw texts
    for sentence in doc['sents']: 
        s_text = " ".join(sentence)
        draw.text((x, sy), s_text, fill="black", font=content_font)
        sy += fsize_c + 10  # Line spacing

    # draw entities
    # TODO : different color for each entity type
    color_e = (0, 256, 0, 100)
    for entities in doc['vertexSet']:
        # foreach type of entity
        for entity in entities:
            # foreach entity instance
            idx_start, idx_end = entity['pos'] 
            idx_sent = entity['sent_id']
            txt = " ".join(doc['sents'][idx_sent][idx_start:idx_end])
            txt_before = " ".join(doc['sents'][idx_sent][:idx_start])
            if len(txt_before) > 0:
                txt_before += " "

            (left, top, right, bottom) = content_font.getbbox(txt_before)
            w1 = right - left
            (left, top, right, bottom) = content_font.getbbox(txt)
            w2 = right - left
            
            # coordinates for box around entity text
            y1 = y + idx_sent * (fsize_c + 10)
            y2 = y1 + fsize_c
            x1 = x + w1
            x2 = x1 + w2
            overlay_draw.rectangle([x1, y1, x2, y2], fill=color_e)

    # combine overlay and image
    img = Image.alpha_composite(img, overlay)

    return img


    """
    # draw relations
    for label in doc['labels']:
        # foreach relation label
        # {'r': 'P17', 'h': 2, 't': 9, 'evidence': [0, 4]}
        r = label['r'] # relation
        idx_h = label['h'] # head
        idx_t = label['t'] # tail
        e = label['evidence'] # list of sentence idxs

        # occurances of head and target entities
        hs = doc['vertexSet'][idx_h]
        ts = doc['vertexSet'][idx_t]

        for idx_sent in e:
            # get y coordinate based on sentence index
            y_r = y + idx_sent * (fsize_c + 10)

            # iterate through sentences with (h -r-> t)
            # get head (h) coordinate (x1)
            sent = doc['sents'][idx_sent]
            pos_h = next((entity["pos"] for entity in hs if entity["sent_id"] == idx_sent), None)
            if pos_h == None:
                continue

            txt_before = " ".join(sent[:pos_h[0]])
            if len(txt_before) > 0:
                txt_before += " "
            (left, top, right, bottom) = content_font.getbbox(txt_before)
            w1 = right - left
            x1 = x + w1

            # get tail (t) coordinate (x2)
            pos_t = next((entity["pos"] for entity in ts if entity["sent_id"] == idx_sent), None)
            if pos_t == None:
                continue

            txt_before = " ".join(sent[:pos_t[0]])
            if len(txt_before) > 0:
                txt_before += " "
            (left, top, right, bottom) = content_font.getbbox(txt_before)
            w2 = right - left
            x2 = x + w2

            # draw lines
            overlay_draw.line((x1, y_r, x1, y_r-5), fill='green', width=3)
            overlay_draw.line((x2, y_r, x2, y_r-5), fill='red', width=3)
    """




    
