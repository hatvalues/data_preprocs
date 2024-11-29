from data_preprocs.data_loading import DataFramework, DataProviderFactory

mush_common_args = {
    "name": "mush",
    "file_name": "mush.csv.gz",
    "sample_size": 1.0,
    "schema": None,
    "class_col": "edible",
    "positive_class": "e",
    "spiel": """Source:
    This dataset was taken from the UCI Machine Learning Repository. The data was donated by Jeff Schlimmer.

    Data Set Information:
    This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like ``leaflets three, let it be'' for Poisonous Oak and Ivy.

    Attribute Information:
    1. cshape (cap-shape): bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
    2. csurface (cap-surface): fibrous=f, grooves=g, scaly=y, smooth=s
    3. ccolor (cap-color): brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
    4. bruises: bruises=t, no=f
    5. odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
    6. gattach (gill-attachment): attached=a, descending=d, free=f, notched=n
    7. gspace (gill-spacing): close=c, crowded=w, distant=d
    8. gsize (gill-size): broad=b, narrow=n
    9. gcolor (gill-color): black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y
    10. shape (stalk-shape): enlarging=e, tapering=t
    11. sroot (stalk-root): bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
    12. ssurfaring (stalk-surface-above-ring): fibrous=f, scaly=y, silky=k, smooth=s
    13. ssurfbring (stalk-surface-below-ring): fibrous=f, scaly=y, silky=k, smooth=s
    14. scoloraring (stalk-color-above-ring): brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
    15. scolorbring (stalk-color-below-ring): brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
    16. vtype (veil-type): partial=p, universal=u
    17. vcolor (veil-color): brown=n, orange=o, white=w, yellow=y
    18. rnum (ring-number): none=n, one=o,
    19. type (ring-type): cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z
    20. sporecolor (spore-print-color): black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y
    21. pop (population): abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y
    22. hab (habitat): grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d
    """,
}

factory = DataProviderFactory(kwargs=mush_common_args | {"data_framework": DataFramework.PANDAS})
mush_pd = factory.create_data_provider()

factory = DataProviderFactory(kwargs=mush_common_args | {"data_framework": DataFramework.POLARS})
mush_pl = factory.create_data_provider()
