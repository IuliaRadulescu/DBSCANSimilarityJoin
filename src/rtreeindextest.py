import rtree.index
 
idx2 = rtree.index.Rtree()

#left, bottom, right, top
 
locs = [
    (14, 10, 14, 10),
    (16, 10, 16, 10),
]
 
for i, (minx, miny, maxx, maxy) in enumerate(locs):
    idx2.add(i, (minx, miny, maxx, maxy), obj={'a': 42})
 
for distance in (1, 2):
    print("Within distance of: ({0})".format(distance))
    print('')
 
    r = [
        (i.id, i.object) 
        for i 
        in idx2.nearest((13, 10, 13, 10), distance, objects=True)
    ]
 
    print(r)
    print('')