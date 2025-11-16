from constants import *

# You should already have these constants defined somewhere:
# BATTLE_ID = 'battle_id'
# TARGET = 'winner'  # or whatever your label key is

BASE_STATS = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']

def _get_stats_from_db(name: str, p_db: dict) -> Dict[str, float]:
    """Internal helper to safely get stats from the pokemon_db."""
    stats = {stat: 0.0 for stat in BASE_STATS}
    db_entry = p_db.get(name)
    if db_entry:
        for stat in BASE_STATS:
            stat_val = db_entry.get(stat, 0.0)
            stats[stat] = float(stat_val) if isinstance(stat_val, (int, float)) else 0.0
    return stats

def extract_static_features_v1(battle: Dict[str, Any], p_db: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts Snapshot 1 (Turn 0) features.
    - P1: Mean stats of all 6 team members.
    - P2: Stats of the single lead Pokémon.
    """
    features = {}

    # P1 Mean Stats (Full 6-Pokémon Team)
    p1_team_stats = []
    for pokemon in battle.get('p1_team_details', []):
        p1_team_stats.append(_get_stats_from_db(pokemon.get('name'), p_db))

    if p1_team_stats:
        p1_stats_df = pd.DataFrame(p1_team_stats)
        for stat in BASE_STATS:
            features[f'p1_mean_{stat}'] = p1_stats_df[stat].mean()
    else:
        for stat in BASE_STATS:
            features[f'p1_mean_{stat}'] = 0.0

    # P2 Lead Stats (Only the revealed lead)
    p2_lead = battle.get('p2_lead_details')
    if p2_lead:
        p2_lead_stats = _get_stats_from_db(p2_lead.get('name'), p_db)
        for stat in BASE_STATS:
            features[f'p2_lead_{stat}'] = p2_lead_stats[stat]
    else:
        for stat in BASE_STATS:
            features[f'p2_lead_{stat}'] = 0.0

    return features

def _get_turn_30_snapshot(battle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Original helper: Process the timeline and find the final state
    of all known Pokémon. Kept as-is for backward compatibility.
    """
    # 1. Initialize P1's "state board" from the full 6-pokemon team
    p1_team_names = {p.get('name') for p in battle.get('p1_team_details', []) if p.get('name')}
    p1_last_state = {name: {'hp_pct': 1.0, 'status': 'nostatus', 'boosts': {}} for name in p1_team_names}

    # 2. Initialize P2's "state board" starting with only the lead
    p2_last_state = {}
    p2_seen_names = set()

    p2_lead = battle.get('p2_lead_details')
    if p2_lead and p2_lead.get('name'):
        p2_lead_name = p2_lead.get('name')
        p2_seen_names.add(p2_lead_name)
        p2_last_state[p2_lead_name] = {'hp_pct': 1.0, 'status': 'nostatus', 'boosts': {}}

    # 3. Process the timeline, updating the "state boards"
    for turn_state in battle.get('battle_timeline', []):
        # P1 state update
        p1_state = turn_state.get('p1_pokemon_state')
        if p1_state and p1_state.get('name') in p1_last_state:
            p1_last_state[p1_state['name']] = p1_state  # Overwrite with the latest state

        # P2 state update (and dynamic discovery)
        p2_state = turn_state.get('p2_pokemon_state')
        if p2_state and p2_state.get('name'):
            p2_name = p2_state['name']
            p2_seen_names.add(p2_name)  # Add to seen set
            p2_last_state[p2_name] = p2_state  # Overwrite with the latest state

    # 4. Return the final states and the count of seen P2 Pokémon
    return {
        'p1_final_states': p1_last_state,
        'p2_final_states': p2_last_state,
        'p2_seen_count': len(p2_seen_names)
    }

def extract_dynamic_features_v2(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts Snapshot 2 features: Fainted counts, HP levels,
    and the P2 unseen count.
    """
    features = {}
    p1_states = snapshot['p1_final_states']
    p2_states = snapshot['p2_final_states']

    # 1. Fainted Counts
    p1_fainted = sum(1 for state in p1_states.values() if state.get('status') == 'fnt')
    p2_fainted = sum(1 for state in p2_states.values() if state.get('status') == 'fnt')

    features['p1_fainted_count'] = p1_fainted
    features['p2_fainted_count'] = p2_fainted
    features['fainted_advantage'] = p2_fainted - p1_fainted  # Higher is better for P1

    # 2. HP Levels (Sum of HP% for non-fainted Pokémon)
    p1_total_hp = sum(state.get('hp_pct', 0.0) for state in p1_states.values() if state.get('status') != 'fnt')
    p2_total_hp = sum(state.get('hp_pct', 0.0) for state in p2_states.values() if state.get('status') != 'fnt')

    # 3. P2 Unseen Pokémon
    p2_seen_count = snapshot['p2_seen_count']
    p2_unseen_count = 6 - p2_seen_count  # Max 6 pokemon per team

    # Assume unseen Pokémon have 100% HP (1.0)
    p2_total_hp_estimated = p2_total_hp + p2_unseen_count

    features['p1_total_hp_pct'] = p1_total_hp
    features['p2_total_hp_pct_estimated'] = p2_total_hp_estimated
    features['hp_advantage'] = p1_total_hp - p2_total_hp_estimated
    features['p2_unseen_count'] = p2_unseen_count

    return features

def extract_other_dynamic_features_v3(snapshot: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts Iteration 3 (REFINED) features: Granular status conditions
    and separated stat boosts from the snapshot.
    """
    features = {}
    p1_states = snapshot['p1_final_states']
    p2_states = snapshot['p2_final_states']

    # === 1. REFINED Status Conditions ===
    INCAPACITATED = {'slp', 'frz'}
    MAJOR_DEBUFF = {'par', 'brn'}
    PASSIVE_DAMAGE = {'psn', 'tox'}

    status_groups = {
        'incapacitated': INCAPACITATED,
        'major_debuff': MAJOR_DEBUFF,
        'passive_damage': PASSIVE_DAMAGE
    }

    # Initialize counts
    for side_prefix in ['p1', 'p2']:
        for group_name in status_groups:
            features[f'{side_prefix}_{group_name}_count'] = 0

    # Count P1 statuses
    for state in p1_states.values():
        if state.get('status') != 'fnt':
            for group_name, status_set in status_groups.items():
                if state.get('status') in status_set:
                    features[f'p1_{group_name}_count'] += 1

    # Count P2 statuses
    for state in p2_states.values():
        if state.get('status') != 'fnt':
            for group_name, status_set in status_groups.items():
                if state.get('status') in status_set:
                    features[f'p2_{group_name}_count'] += 1

    # === 2. REFINED Stat Boosts ===
    p1_boosts = {'off': 0, 'def': 0, 'spe': 0}
    p2_boosts = {'off': 0, 'def': 0, 'spe': 0}

    # Sum P1 boosts
    for state in p1_states.values():
        if state.get('status') != 'fnt':
            boost_dict = state.get('boosts', {})
            p1_boosts['off'] += boost_dict.get('atk', 0) + boost_dict.get('spa', 0)
            p1_boosts['def'] += boost_dict.get('def', 0) + boost_dict.get('spd', 0)
            p1_boosts['spe'] += boost_dict.get('spe', 0)

    # Sum P2 boosts
    for state in p2_states.values():
        if state.get('status') != 'fnt':
            boost_dict = state.get('boosts', {})
            p2_boosts['off'] += boost_dict.get('atk', 0) + boost_dict.get('spa', 0)
            p2_boosts['def'] += boost_dict.get('def', 0) + boost_dict.get('spd', 0)
            p2_boosts['spe'] += boost_dict.get('spe', 0)

    # Create final boost features
    features['p1_offensive_boost_sum'] = p1_boosts['off']
    features['p2_offensive_boost_sum'] = p2_boosts['off']
    features['offensive_boost_advantage'] = p1_boosts['off'] - p2_boosts['off']

    features['p1_defensive_boost_sum'] = p1_boosts['def']
    features['p2_defensive_boost_sum'] = p2_boosts['def']
    features['defensive_boost_advantage'] = p1_boosts['def'] - p2_boosts['def']

    features['p1_speed_boost_sum'] = p1_boosts['spe']
    features['p2_speed_boost_sum'] = p2_boosts['spe']
    features['speed_boost_advantage'] = p1_boosts['spe'] - p2_boosts['spe']

    return features

# ---------------------------------------------------------------------
# ORIGINAL (non-sequential) pipeline — unchanged
# ---------------------------------------------------------------------
def create_features(data: List[Dict[str, Any]],
                    p_db: Dict[str, Any],
                    iteration_name: str) -> (pd.DataFrame, pd.Series):
    """
    Main pipeline to create a feature DataFrame based on the iteration name.

    - v1: Static features only.
    - v2: v1 + Dynamic HP/Fainted features.
    - v3: v2 + Dynamic Status/Boost features.
    """
    print(f"\n--- Creating features for '{iteration_name}' --- ")
    feature_list = []
    labels = []

    if not p_db:
        raise ValueError("Pokemon DB is empty! Please re-run Section 2.3.")

    for battle in tqdm(data, desc=f"Processing '{iteration_name}' features"):
        features = {}

        # --- Snapshot 1 (Turn 0) ---
        if iteration_name in ['v1', 'v2', 'v3']:
            features.update(extract_static_features_v1(battle, p_db))

        # --- Snapshot 2 (Turn 30) ---
        if iteration_name in ['v2', 'v3']:
            snapshot_30 = _get_turn_30_snapshot(battle)
            features.update(extract_dynamic_features_v2(snapshot_30))

        # --- Refinements (Other Dynamics) ---
        if iteration_name in ['v3']:
            features.update(extract_other_dynamic_features_v3(snapshot_30))

        # --- Add Metadata ---
        features[BATTLE_ID] = battle.get(BATTLE_ID)
        if TARGET in battle:
            labels.append(int(battle[TARGET]))

        feature_list.append(features)

    # Create DataFrame
    X = pd.DataFrame(feature_list).fillna(0)
    y = pd.Series(labels, name=TARGET) if labels else None

    print(f"Successfully created feature set with {X.shape[1] - 1} features.")

    # Separate features from metadata
    feature_cols = [col for col in X.columns if col not in [TARGET, BATTLE_ID]]

    return X[feature_cols], y

print("Defined main 'create_features' pipeline function (v1, v2, v3 enabled).")

# ---------------------------------------------------------------------
# NEW: per-turn sequential feature extractor
# ---------------------------------------------------------------------
def extract_sequential_features(
    battle: Dict[str, Any],
    p_db: Dict[str, Any],
    iteration_name: str,
    max_turns: int = None
) -> List[Dict[str, float]]:
    """
    Extract a sequence of per-turn feature vectors for a sequential model.

    For each turn t in battle['battle_timeline'], we build:
      - Turn index: 'turn_index'
      - v1: Static features (constant across turns)
      - v2: Dynamic HP/Fainted/Unseen features at turn t
      - v3: Dynamic status/boost features at turn t

    iteration_name controls which subsets are included:
      'v1' -> static only
      'v2' -> static + v2
      'v3' -> static + v2 + v3

    Returns:
        List[Dict[str, float]] of length = number of turns (or max_turns),
        where each dict is the feature vector at that turn.
    """

    # -------- 1. Static features (constant across turns) --------
    static_feats: Dict[str, float] = {}
    if iteration_name in ['v1', 'v2', 'v3']:
        static_feats = extract_static_features_v1(battle, p_db)

    # -------- 2. Initialize state boards similarly to _get_turn_30_snapshot --------
    p1_team_names = {p.get('name') for p in battle.get('p1_team_details', []) if p.get('name')}
    p1_last_state = {name: {'hp_pct': 1.0, 'status': 'nostatus', 'boosts': {}} for name in p1_team_names}

    p2_last_state = {}
    p2_seen_names = set()

    p2_lead = battle.get('p2_lead_details')
    if p2_lead and p2_lead.get('name'):
        p2_lead_name = p2_lead.get('name')
        p2_seen_names.add(p2_lead_name)
        p2_last_state[p2_lead_name] = {'hp_pct': 1.0, 'status': 'nostatus', 'boosts': {}}

    # -------- 3. Iterate over turns and build per-turn features --------
    sequential_features: List[Dict[str, float]] = []
    timeline = battle.get('battle_timeline', [])

    for turn_idx, turn_state in enumerate(timeline, start=1):
        if max_turns is not None and turn_idx > max_turns:
            break

        # Update P1 state
        p1_state = turn_state.get('p1_pokemon_state')
        if p1_state and p1_state.get('name') in p1_last_state:
            p1_last_state[p1_state['name']] = p1_state

        # Update P2 state + new discoveries
        p2_state = turn_state.get('p2_pokemon_state')
        if p2_state and p2_state.get('name'):
            p2_name = p2_state['name']
            p2_seen_names.add(p2_name)
            p2_last_state[p2_name] = p2_state

        # Build snapshot at this turn
        snapshot = {
            'p1_final_states': p1_last_state,
            'p2_final_states': p2_last_state,
            'p2_seen_count': len(p2_seen_names)
        }

        turn_features: Dict[str, float] = {}
        turn_features['turn_index'] = float(turn_idx)

        if iteration_name in ['v1', 'v2', 'v3']:
            turn_features.update(static_feats)

        if iteration_name in ['v2', 'v3']:
            dyn_v2 = extract_dynamic_features_v2(snapshot)
            turn_features.update(dyn_v2)

        if iteration_name in ['v3']:
            dyn_v3 = extract_other_dynamic_features_v3(snapshot)
            turn_features.update(dyn_v3)

        sequential_features.append(turn_features)

    return sequential_features

# ---------------------------------------------------------------------
# NEW: main sequential pipeline for RNN/Transformer models
# ---------------------------------------------------------------------
def create_sequential_features(
    data: List[Dict[str, Any]],
    p_db: Dict[str, Any],
    iteration_name: str,
    max_turns: int = 30,
    pad_value: float = 0.0
) -> Tuple[np.ndarray, pd.Series, List[str], np.ndarray]:
    """
    Create a 3D feature tensor suitable for sequential models.

    Args:
        data: list of battle dicts.
        p_db: pokemon DB.
        iteration_name: 'v1', 'v2', or 'v3' (same semantics as create_features).
        max_turns: pad/truncate each battle to this many turns.
        pad_value: value used for padding shorter battles.

    Returns:
        X_seq: np.ndarray of shape (n_battles, max_turns, n_features)
        y:    pd.Series of labels (or None if labels absent)
        feature_names: list of feature names for last dimension of X_seq
        lengths: np.ndarray of shape (n_battles,), actual (clipped) lengths
                 before padding for each battle.
    """
    print(f"\n--- Creating SEQUENTIAL features for '{iteration_name}' --- ")

    if not p_db:
        raise ValueError("Pokemon DB is empty! Please re-run Section 2.3.")

    all_battle_arrays: List[np.ndarray] = []
    labels: List[int] = []
    lengths_list: List[int] = []
    feature_cols: List[str] = None

    for battle in tqdm(data, desc=f"Processing '{iteration_name}' sequential features"):
        # Per-turn dicts
        seq_feat_dicts = extract_sequential_features(
            battle, p_db, iteration_name=iteration_name, max_turns=max_turns
        )

        # If battle has no turns, we still want a padded row of pad_value
        if not seq_feat_dicts:
            # create a dummy row with just turn_index; columns will be aligned later
            seq_feat_dicts = [{'turn_index': 1.0}]

        df_turns = pd.DataFrame(seq_feat_dicts).fillna(0)

        # Initialize / check global feature columns
        if feature_cols is None:
            feature_cols = list(df_turns.columns)
        else:
            # Ensure consistent columns across battles
            new_cols = [c for c in df_turns.columns if c not in feature_cols]
            if new_cols:
                raise ValueError(f"Inconsistent feature columns across battles, new columns: {new_cols}")
            df_turns = df_turns.reindex(columns=feature_cols, fill_value=0.0)

        # Pad or truncate to max_turns
        num_turns = len(df_turns)
        actual_len = min(num_turns, max_turns)
        lengths_list.append(actual_len)

        if num_turns >= max_turns:
            df_turns = df_turns.iloc[:max_turns, :]
            seq_array = df_turns.to_numpy(dtype=float)
        else:
            seq_array = df_turns.to_numpy(dtype=float)
            pad_rows = max_turns - num_turns
            if pad_rows > 0:
                pad_block = np.full((pad_rows, seq_array.shape[1]), pad_value, dtype=float)
                seq_array = np.vstack([seq_array, pad_block])

        all_battle_arrays.append(seq_array)

        # Labels
        if TARGET in battle:
            labels.append(int(battle[TARGET]))

    X_seq = np.stack(all_battle_arrays, axis=0)  # (n_battles, max_turns, n_features)
    y = pd.Series(labels, name=TARGET) if labels else None
    lengths = np.array(lengths_list, dtype=int)

    print(f"Sequential feature tensor shape: {X_seq.shape} "
          f"(battles, max_turns, features={X_seq.shape[-1]})")

    return X_seq, y, feature_cols, lengths

print("Defined per-turn sequential feature extraction and create_sequential_features().")
