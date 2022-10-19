import heapq
from collections import defaultdict, Counter
import random
import gzip
import sys
import numpy as np
import pandas as pd
import csv

def track_id2freebase_id():
    valid_items_dataset = [int(pid) for pid in list(pd.read_csv("LFM1M/products.txt", sep="\t").pid.unique())]
    valid_items_kg_df = pd.read_csv("LFM-1b/lfb2fb.txt", sep="\t", names=["pid", "freebase_id"])
    valid_items_kg_df = valid_items_kg_df[valid_items_kg_df.pid.isin(valid_items_dataset)]
    return dict(zip(valid_items_kg_df.pid, valid_items_kg_df.freebase_id))

def remove_duplicates():
    pass

def remove_users_without_sensible_attribute():
    pass

def remove_invalid_reviews():
    df = pd.read_csv("LFM-1b/LFM-1b_no_duplicates.txt", sep="\t", names=["uid", "pid", "feedback", "timestamp"],
                     header=None)

    #Invalid because of user without sensible attributes
    print("Removing reviews involving invalid user")
    df_users = pd.read_csv("LFM-1b/LFM-1b_users.txt", sep="\t")
    df_users = df_users.drop(df_users[((df_users.gender != "m") & (df_users.gender != "f")) | (
                (df_users.age < 0) | (df_users.age > 100))].index)
    df = df.drop(df[~df["uid"].isin(df_users.user_id)].index)

    print("Removing reviews involving invalid items")
    to_kg_df = pd.read_csv("LFM-1b/lfb2fb.txt", sep="\t", names=["pid", "kg_id"])
    df = df.drop(df[~df["pid"].isin(to_kg_df["pid"])].index)

    print("Performing k-core")
    counts_col_user = df.groupby("uid")["uid"].transform('count')
    counts_col_songs = df.groupby("pid")["pid"].transform('count')
    k_user, k_song = 20, 10
    mask_user = counts_col_user >= k_user
    mask_songs = counts_col_songs >= k_song
    df = df[mask_user & mask_songs]
    df.to_csv("LFM-1b/LFM-filtered.txt", sep="\t", index=False)

def create_representative_sample():
    df = pd.read_csv("LFM-1b/LFM-filtered.txt", sep="\t")
    df_users = pd.read_csv("LFM-1b/LFM-1b_users.txt", sep="\t")
    df_users.set_axis(["uid", "country", "age", "gender", "playcount", "registered_unixtime"], axis=1, inplace=True)
    df_users = df_users.drop(df_users[~df_users.uid.isin(df.uid)].index)
    df_users = df_users.drop(["country", "playcount", "registered_unixtime"], axis=1)
    df = df.merge(df_users, on="uid")
    print("---Stats LFM filtered---")
    print("Number of male:", df_users[df_users.gender.isin(["m"])].shape[0])
    print("Number of female:", df_users[~df_users.gender.isin(["m"])].shape[0])
    print(f"Male interactions: {df[df.gender.isin(['m'])].shape[0]}")
    print(f"Female interactions: {df[~df.gender.isin(['m'])].shape[0]}")
    print("Avg interactions per male:",
          df[df.gender.isin(["m"])].shape[0] / df_users[df_users.gender.isin(["m"])].shape[0])
    print("Avg interactions per female:",
          df[~df.gender.isin(["m"])].shape[0] / df_users[~df_users.gender.isin(["m"])].shape[0])
    print(f"Number of items: {df.pid.unique().shape[0]}")
    print("Avg occurence per item: ",
          df.shape[0] / df.pid.unique().shape[0])

    no_of_male_users, no_of_female_users = df_users[df_users.gender.isin(["m"])].shape[0], df_users[~df_users.gender.isin(["m"])].shape[0]
    male_interactions, female_interactions = df[df.gender.isin(['m'])].shape[0], df[~df.gender.isin(['m'])].shape[0]
    percentage_males = male_interactions / df.shape[0]
    percentage_females = female_interactions / df.shape[0]

    user_item_ratio = 2.5 #Original is 20 too sparse
    k_user = 60
    print(f"---Stats LFM user k-core {k_user}---")
    counts_col_user = df.groupby("uid")["uid"].transform('count')
    mask_user = counts_col_user >= k_user
    df = df[mask_user]
    df_users = df_users.drop(df_users[~df_users.uid.isin(df.uid)].index)
    no_of_male_users, no_of_female_users = df_users[df_users.gender.isin(["m"])].shape[0], df_users[~df_users.gender.isin(["m"])].shape[0]
    male_interactions, female_interactions = df[df.gender.isin(['m'])].shape[0], df[~df.gender.isin(['m'])].shape[0]
    print("Number of male:", no_of_male_users)
    print("Number of female:", no_of_female_users)
    print(f"Male interactions: {male_interactions}")
    print(f"Female interactions: {female_interactions}")
    print("Avg interactions per male:",
          df[df.gender.isin(["m"])].shape[0] / no_of_male_users)
    print("Avg interactions per female:",
          df[~df.gender.isin(["m"])].shape[0] / no_of_female_users)
    print(f"Number of items: {df.pid.unique().shape[0]}")
    print("Avg occurence per item: ",
          df.shape[0] / df.pid.unique().shape[0])

    sample_size_users = 5000
    male_uids = list(df[df.gender.isin(["m"])].uid.unique())
    female_uids = list(df[~df.gender.isin(["m"])].uid.unique())

    random.seed(2022)
    sampled_males = random.sample(male_uids, int(sample_size_users*percentage_males))
    sampled_females = random.sample(female_uids, int(sample_size_users*percentage_females))
    df = df[df.uid.isin(sampled_males) | df.uid.isin(sampled_females)]

    #K-core items
    k_pid = 40
    counts_col_item = df.groupby("pid")["pid"].transform('count')
    mask_item = counts_col_item >= k_pid
    df = df[mask_item]
    valid_items = list(df.pid.unique())
    sampled_items = random.sample(valid_items, int(sample_size_users*user_item_ratio))
    df = df[df.pid.isin(sampled_items)]

    print(f"---Stats sample---")
    df_users = df_users.drop(df_users[~df_users.uid.isin(df.uid)].index)
    no_of_male_users, no_of_female_users = df_users[df_users.gender.isin(["m"])].shape[0], \
                                           df_users[~df_users.gender.isin(["m"])].shape[0]
    male_interactions, female_interactions = df[df.gender.isin(['m'])].shape[0], df[~df.gender.isin(['m'])].shape[0]
    print("Number of users", no_of_male_users + no_of_female_users)
    print("Number of male:", no_of_male_users)
    print("Number of female:", no_of_female_users)
    print(f"Number of interactions: {male_interactions + female_interactions}")
    print(f"Male interactions: {male_interactions}")
    print(f"Female interactions: {female_interactions}")
    print("Avg interactions per male:",
          df[df.gender.isin(["m"])].shape[0] / no_of_male_users)
    print("Avg interactions per female:",
          df[~df.gender.isin(["m"])].shape[0] / no_of_female_users)
    print(f"Number of items: {df.pid.unique().shape[0]}")
    print("Avg occurence per item: ",
          df.shape[0] / df.pid.unique().shape[0])

    df = df.drop(["gender", "age"], axis=1)
    df.to_csv("LFM1M/ratings.txt", sep="\t", header=True, index=False)

def creare_users_item_from_ratings():
    df = pd.read_csv("LFM1M/ratings.txt", sep="\t")

    df_users = pd.read_csv("LFM-1b/LFM-1b_users.txt", sep="\t")
    df_users.set_axis(["uid", "country", "age", "gender", "playcount", "registered_unixtime"], axis=1, inplace=True)
    df_users = df_users.drop(df_users[~df_users.uid.isin(df.uid)].index)
    df_users = df_users.drop(["country", "playcount", "registered_unixtime"], axis=1)

    print("Number of male:", df_users[df_users.gender.isin(["m"])].shape[0])
    print("Number of female:", df_users[~df_users.gender.isin(["m"])].shape[0])
    df = df.merge(df_users, on="uid")
    print(f"Male interactions: {df[df.gender.isin(['m'])].shape[0]}")
    print(f"Female interactions: {df[~df.gender.isin(['m'])].shape[0]}")
    print("Avg interactions per male:",
          df[df.gender.isin(["m"])].shape[0] / df_users[df_users.gender.isin(["m"])].shape[0])
    print("Avg interactions per female:",
          df[~df.gender.isin(["m"])].shape[0] / df_users[~df_users.gender.isin(["m"])].shape[0])

    df_users.to_csv("LFM1M/users.txt", sep="\t", columns=["uid", "age", "gender"], index=False)

    print(f"No of interactions: {df.shape[0]}")
    print(f"No of users: {df_users.shape[0]}")
    df_items = pd.read_csv("LFM-1b/LFM-1b_tracks.txt", sep="\t", names=["pid", "name", "artist_id"])
    df_items = df_items.drop(df_items[~df_items.pid.isin(df.pid)].index)
    df = df.drop(df[~df.pid.isin(df_items.pid)].index)
    df.drop(["age", "gender"], axis=1, inplace=True)
    df.to_csv("LFM1M/ratings.txt", sep="\t", index=False)
    df_items.to_csv("LFM1M/products.txt", sep="\t", columns=["pid", "name", "artist_id"], index=False)
    print(f"No of items: {df_items.shape[0]}")

def create_track2gendre():
    df_tracks = pd.read_csv("LFM1M/products.txt", sep="\t")
    valid_artists = [int(id) for id in list(df_tracks["artist_id"].unique())]
    df_artist = pd.read_csv("LFM-1b/LFM-1b_artists.txt", sep="\t", names=["id", "name"])
    df_artist = df_artist[df_artist.id.isin(valid_artists)]
    artist_name2id = dict(zip(df_artist.name, df_artist.id))

    artist2genre_rows = []
    with open("LFM-1b/LFM-1b_artist_genres_freebase.txt", 'r') as artist2genre_raw_file:
        reader = csv.reader(artist2genre_raw_file, delimiter="\t")
        for row in reader:
            if row[0] not in artist_name2id:
                continue
            artist = artist_name2id[row[0]]
            genre = row[1]
            artist2genre_rows.append([artist, genre])
    artist2genre_raw_file.close()

    artist2genre_df = pd.DataFrame(artist2genre_rows, columns=["artist_id", "genre"])
    df_tracks = pd.merge(df_tracks, artist2genre_df, how="left", on="artist_id")
    df_tracks.drop_duplicates(subset="pid", inplace=True)
    df_tracks.to_csv("LFM1M/products.txt", sep="\t", index=False)

def add_genre_triplets():
    df_tracks = pd.read_csv("LFM1M/products.txt", sep="\t")
    df_tracks.drop(["genre"], axis=1, inplace=True)
    valid_artists = [int(id) for id in list(df_tracks["artist_id"].unique())]
    df_artist = pd.read_csv("LFM-1b/LFM-1b_artists.txt", sep="\t", names=["id", "name"])
    df_artist = df_artist[df_artist.id.isin(valid_artists)]
    artist_name2id = dict(zip(df_artist.name, df_artist.id))

    artist2genre_rows = []
    with open("LFM-1b/LFM-1b_artist_genres_freebase.txt", 'r') as artist2genre_raw_file:
        reader = csv.reader(artist2genre_raw_file, delimiter="\t")
        for row in reader:
            if row[0] not in artist_name2id:
                continue
            artist = artist_name2id[row[0]]
            genres = [genre for genre in row[1:] if len(genre) > 0]
            artist2genre_rows.append([artist, genres])
    artist2genre_raw_file.close()

    artist2genre_df = pd.DataFrame(artist2genre_rows, columns=["artist_id", "genres"])
    df_tracks = df_tracks.merge(artist2genre_df, on="artist_id")
    track2genres = dict(zip(df_tracks.pid, df_tracks.genres))
    new_triplet_rows = []
    track2kg_id = track_id2freebase_id()
    for pid, genres in track2genres.items():
        for genre in genres:
            new_triplet_rows.append([track2kg_id[pid], "<http://rdf.freebase.com/ns/music.recording.genre>",genre])

    df_triplets_genre = pd.DataFrame(new_triplet_rows, columns=["entity_head", "relation", "entity_tail"])
    df_triplets = pd.read_csv("LFM-1b/triplets-filtered-processed.txt", sep='\t')
    df_triplets = pd.concat([df_triplets, df_triplets_genre], ignore_index=True)
    df_triplets.to_csv("LFM-1b/triplets-final.txt", sep="\t", index=False)

def crawl_triplets_freebase():
    valid_items_dataset = [int(pid) for pid in list(pd.read_csv("LFM1M/products.txt", sep="\t").pid.unique())]
    valid_items_kg_df = pd.read_csv("LFM-1b/lfb2fb.txt", sep="\t", names=["pid", "freebase_id"])
    valid_items_kg_df = valid_items_kg_df[valid_items_kg_df.pid.isin(valid_items_dataset)]
    valid_items_kg = set(valid_items_kg_df.freebase_id.unique())

    valid_relations = set()
    with gzip.open("LFM-1b/triplets-filtered.txt.gz", 'wt') as triplets_file:
        writer = csv.writer(triplets_file, delimiter="\t")
        with gzip.open("LFM-1b/freebase-rdf-latest.gz", 'rt') as kg_file:
            def fix_nulls(s):
                for line in s:
                    yield line.replace('\0', ' ')
            reader = csv.reader(fix_nulls(kg_file), delimiter="\t")

            for row in reader:
                entity_head, relation, entity_tail = row[:3]
                entity_tail = entity_tail.split("/")[-1][:-1]
                entity_head = entity_head.split("/")[-1][:-1]
                if (entity_head in valid_items_kg) and (entity_tail not in valid_items_kg_df):
                    writer.writerow([entity_head, relation, entity_tail])
                    valid_relations.add(relation)
                elif (entity_head not in valid_items_kg) and (entity_tail in valid_items_kg_df):
                    writer.writerow([entity_tail, relation, entity_head])
                    valid_relations.add(relation)
        kg_file.close()
    triplets_file.close()

    with open("LFM1M/relations.txt", 'w+') as r_file:
        writer = csv.writer(r_file, delimiter="\t")
        for rel in valid_relations:
            writer.writerow([rel])
    r_file.close()

def extract_relations():
    valid_items_dataset = [int(pid) for pid in list(pd.read_csv("LFM1M/products.txt", sep="\t").pid.unique())]
    valid_items_kg_df = pd.read_csv("LFM-1b/lfb2fb.txt", sep="\t", names=["pid", "freebase_id"])
    valid_items_kg_df = valid_items_kg_df[valid_items_kg_df.pid.isin(valid_items_dataset)]
    valid_items_kg = set(valid_items_kg_df.freebase_id.unique())

    df = pd.read_csv("LFM-1b/triplets-filtered.txt", names=["entity_head", "relation", "entity_tail"], sep='\t')
    df = df.drop(df[(df.entity_head.isin(valid_items_kg)) & (df.entity_tail.isin(valid_items_kg))].index)
    v = df[['relation']]
    df = df[v.replace(v.apply(pd.Series.value_counts)).gt(300).all(1)]
    df.to_csv("LFM-1b/triplets-filtered-processed.txt", sep="\t", index=False)
    df = pd.DataFrame(df.relation.unique())
    df.to_csv("LFM-1b/relations_forward.txt", sep="\t", columns=["kb_relation"], index=False)
    df = pd.read_csv("LFM-1b/triplets-filtered.txt", names=["entity_head", "relation", "entity_tail"], sep='\t')
    df = df[df.entity_tail.isin(valid_items_kg)]
    v = df[['relation']]
    df = df[v.replace(v.apply(pd.Series.value_counts)).gt(300).all(1)]
    df = pd.DataFrame(df.relation.unique())
    df.to_csv("LFM-1b/relations_backwords.txt", columns=["kb_relation"], sep="\t", index=False)

def filter_entities():
    df = pd.read_csv("LFM-1b/triplets-filtered-processed.txt", names=["entity_head", "relation", "entity_tail"], sep='\t')
    df_valid_tails = df.entity_tail.unique()
    entity_df = pd.read_csv("LFM-1b/e_map_raw.txt", names=["entity", "name"], sep='\t')
    entity_df = entity_df[entity_df.entity.isin(df_valid_tails)]
    entity_df = entity_df[entity_df.name.str.contains("@en")]
    if entity_df.shape[0] == 0:
        return
    entity_df.name = entity_df.name.apply(lambda x: x.split("@")[0])
    entity_df.to_csv("LFM-1b/e_map_raw.txt", sep="\t", index=False)

def create_emap():
    df = pd.read_csv("LFM-1b/triplets-filtered-processed.txt", names=["entity_head", "relation", "entity_tail"], sep='\t')
    #df_valid_heads = df.entity_head.unique()
    dataset_id2kg_entity = pd.read_csv("LFM-1b/lfb2fb.txt", names=["pid", 'entity'], sep="\t")
    products_df = pd.read_csv("LFM1M/products.txt", sep="\t")
    products_df = products_df[["pid", "name"]]
    products_df = products_df.merge(dataset_id2kg_entity, on="pid")
    products_df.drop_duplicates(inplace=True)
    entity_tail_df = df[["entity_tail"]]
    entity_tail_df.drop_duplicates(inplace=True)
    entity_df = pd.read_csv("LFM-1b/e_map_raw.txt", sep='\t')
    entity_df = pd.merge(entity_tail_df, entity_df,  left_on="entity_tail", right_on="entity")
    entity_df.drop_duplicates(subset="entity", inplace=True)
    entity_df.drop(["entity_tail"], axis=1, inplace=True)
    entity_df.fillna("Name not found", inplace=True)
    i2kg = products_df.copy()
    i2kg.drop_duplicates(inplace=True)
    products_df.drop(["pid"], axis=1, inplace=True)
    entity_df.drop_duplicates(inplace=True)
    #Add genre entities
    genre_df = pd.read_csv("LFM-1b/genres_freebase.txt", names=["name"], sep="\t")
    genre_df.insert(0, 'entity', range(genre_df.shape[0]))  # Create a new incremental ID
    entity_df = pd.concat([products_df, entity_df, genre_df], ignore_index=True)
    entity_df.insert(0, 'eid', range(entity_df.shape[0]))  # Create a new incremental ID
    entity_df.to_csv("LFM1M/kg/e_map.txt", sep="\t", index=False)
    i2kg.insert(0, 'eid', range(i2kg.shape[0]))
    i2kg.to_csv("LFM1M/kg/i2kg_map.txt", sep="\t", index=False)

def remove_invalid_relation_triplets():
    df = pd.read_csv("LFM-1b/triplets-filtered.txt", names=["entity_head", "relation", "entity_tail"], sep='\t')
    df_valid_rel_forward = pd.read_csv("LFM-1b/relations_forward.txt", names=["relation"], sep='\t')
    df = df[df.relation.isin(df_valid_rel_forward.relation)]
    df.to_csv("LFM-1b/triplets-filtered-processed.txt", sep="\t", index=False)

def remove_invalid_entity_triplets():
    df = pd.read_csv("LFM-1b/triplets-filtered-processed.txt", names=["entity_head", "relation", "entity_tail"],
                     sep='\t')
    valid_products = pd.read_csv("LFM1M/products.txt", sep="\t")
    dataset_id2kg_entity = pd.read_csv("LFM-1b/lfb2fb.txt", names=["pid", 'entity'], sep="\t")
    dataset_id2kg_entity = dataset_id2kg_entity[dataset_id2kg_entity.pid.isin(valid_products.pid)]
    df = df[df.entity_head.isin(dataset_id2kg_entity.entity)]
    df.to_csv("LFM-1b/triplets-filtered-processed.txt", sep="\t", index=False)

def crawl_entity_names():
    df = pd.read_csv("LFM-1b/triplets-filtered-processed.txt", names=["entity_head", "relation", "entity_tail"], sep='\t')
    df_valid_tails = set(df.entity_tail.unique())

    with open("LFM-1b/e_map_raw.txt", 'w+') as entities_file:
        writer = csv.writer(entities_file, delimiter="\t")
        with gzip.open("LFM-1b/freebase-rdf-latest.gz", 'rt') as kg_file:
            def fix_nulls(s):
                for line in s:
                    yield line.replace('\0', ' ')

            reader = csv.reader(fix_nulls(kg_file), delimiter="\t")
            for row in reader:
                entity_head, relation, entity_tail = row[:3]
                entity_head = entity_head.split("/")[-1][:-1]
                if relation != "<http://rdf.freebase.com/ns/type.object.name>" or entity_head not in df_valid_tails: continue
                writer.writerow([entity_head, entity_tail])
        kg_file.close()
    entities_file.close()
    e_map = pd.read_csv("LFM-1b/e_map_raw.txt", sep="\t", header=["entity", "name"])
    e_map.drop_duplicates(inplace=True)
    e_map.to_csv("LFM-1b/e_map_raw.txt", sep="\t", index=False)

def create_rmap():
    freebase_relation2plain = {
        "<http://rdf.freebase.com/ns/music.recording.artist>": "featured_by_artist",
        "<http://rdf.freebase.com/ns/music.recording.engineer>": "mixed_by_engineer",
        "<http://rdf.freebase.com/ns/music.recording.producer>": "produced_by_producer",
    }
    df_valid_rel_forward = pd.read_csv("LFM-1b/relations_forward.txt", sep='\t', names=["kb_relation"])
    df_valid_rel_forward["name"] = df_valid_rel_forward.kb_relation.map(freebase_relation2plain)
    genre_series = pd.Series(["<http://rdf.freebase.com/ns/music.recording.genre>", "belong_to_genre"], index=df_valid_rel_forward.columns)
    df_valid_rel_forward = df_valid_rel_forward.append(genre_series, ignore_index=True)
    df_valid_rel_forward.insert(0, 'id', range(df_valid_rel_forward.shape[0])) #Create a new incremental ID
    df_valid_rel_forward.to_csv("LFM1M/kg/r_map.txt", sep="\t", index=False)

def create_kg_final():
    df = pd.read_csv("LFM-1b/triplets-final.txt",
                     sep='\t')
    df_valid_rel = pd.read_csv("LFM1M/kg/r_map.txt", sep='\t')
    df_kb_rel2plain_tex = dict(zip(df_valid_rel.kb_relation, df_valid_rel.id))
    df.relation = df.relation.map(df_kb_rel2plain_tex)
    df.to_csv("LFM1M/kg/kg_final.txt", sep="\t", index=False)

def normalize_gender():
    pass

def normalize_age():
    pass

def create_lfm1m_data():
    #remove_invalid_reviews()
    #create_representative_sample()
    creare_users_item_from_ratings()
    create_track2gendre()
    preprocess_age()
    preprocess_gender()

def triplets(crawl=False, relations_cleaned=True):
    if crawl:
        csv.field_size_limit(sys.maxsize)
        crawl_triplets_freebase()
        crawl_entity_names()
    #...
    if not relations_cleaned:
        extract_relations()
    #Manually remove invalid relations

    remove_invalid_relation_triplets()
    remove_invalid_entity_triplets()
    filter_entities()

def create_kg_files():
    add_genre_triplets()
    create_rmap()
    create_emap()
    create_kg_final()

def make_interval(age):
    if age < 18:
        return "Under 18"
    if age >= 18 and age <= 24:
        return "18-24"
    if age >= 25 and age <= 34:
        return "25-34"
    if age >= 35 and age <= 44:
        return "35-44"
    if age >= 45 and age <= 49:
        return "45-49"
    if age >= 50 and age <= 55:
        return "50-55"
    if age >= 56:
        return "56+"

def preprocess_age():
    users_df = pd.read_csv("LFM1M/users.txt", sep="\t")
    users_df.age = users_df.age.apply(lambda x: make_interval(x))
    users_df.to_csv("LFM1M/users.txt", sep="\t", index=False)

def preprocess_gender():
    users_df = pd.read_csv("LFM1M/users.txt", sep="\t")
    gender_map = {"m": "M", "f": "F"}
    users_df.gender = users_df.gender.map(gender_map)
    users_df.to_csv("LFM1M/users.txt", sep="\t", index=False)

def calculate_popularity():
    #Item popularity
    interactions_df = pd.read_csv("LFM1M/ratings.txt", sep="\t")
    product2interaction_number = Counter(interactions_df.pid)
    most_interacted, less_interacted = max(product2interaction_number.values()), min(product2interaction_number.values())
    for pid in product2interaction_number.keys():
        occ = product2interaction_number[pid]
        product2interaction_number[pid] = (occ - less_interacted) /(most_interacted - less_interacted)

    products_df = pd.read_csv("LFM1M/products.txt", sep="\t")
    products_df.insert(3, "pop_item", product2interaction_number.values(), allow_duplicates=True)

    #Provider popularity
    track2artist = dict(zip(products_df.pid, products_df.artist_id))
    interaction_artist_df = interactions_df.copy()
    interaction_artist_df["artist_id"] = interaction_artist_df.pid.map(track2artist)
    provider2interaction_number = Counter(interaction_artist_df.artist_id)
    most_interacted, less_interacted = max(provider2interaction_number.values()), min(provider2interaction_number.values())
    for pid in provider2interaction_number.keys():
        occ = provider2interaction_number[pid]
        provider2interaction_number[pid] = (occ - less_interacted) / (most_interacted - less_interacted)
    products_df["pop_provider"]  = products_df.artist_id.map(provider2interaction_number)
    products_df.to_csv("LFM1M/products.txt", sep="\t", index=False)

#create_lfm1m_data()
#triplets()
preprocess_age()