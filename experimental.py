def detect_bot(revisions):
    user_name = revisions['user_name'].iloc[0]
    if user_name.endswith("Bot"):
        return True
    revisions = revisions.copy()
    revisions["ts_f"] = revisions['timestamp'].dt.floor("1h")
    revisions = revisions.drop_duplicates("ts_f")
    
    return revisions.rolling(on="ts_f", window="24h")["user_id"].count().max() == 24
