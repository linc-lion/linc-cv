import json

linc_images_db = {}

with open('linc_db.json') as f:
    j = json.load(f)

cnt = 0
for i, k in enumerate(j):
    try:
        for imset in k['_embedded']['image_sets']:
            for image in imset['_embedded']['images']:
                t = image['image_type']
                tn = image['thumbnail_url']
                linc_images_db.setdefault(i, {})
                linc_images_db[i].setdefault(t, [])
                linc_images_db[i][t].append(tn)
                print(cnt, str(i), t, tn)
                cnt += 1
    except KeyError:
        continue

with open('linc_images_db.json', 'w') as f:
    json.dump(linc_images_db, f)
