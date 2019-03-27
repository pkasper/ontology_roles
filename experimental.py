def detect_bot(revisions):
    revisions = revisions.copy()
    revisions["ts_f"] = revisions['timestamp'].dt.floor("1h")
    revisions = revisions.drop_duplicates("ts_f")
    
    return revisions.rolling(on="ts_f", window="48h")["user_id"].count().max() == 48
